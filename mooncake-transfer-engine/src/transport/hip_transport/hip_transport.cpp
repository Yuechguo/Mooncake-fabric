// Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "transport/hip_transport/hip_transport.h"

#include <glog/logging.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <map>
#include <unistd.h>
#include <sys/syscall.h>

#include "common.h"
#include "common/serialization.h"
#include "config.h"
#include "hip_device_guard.h"
#include "transfer_metadata.h"
#include "transport/hip_transport/hip_batch_copy.h"
#include "transport/hip_transport/hip_share_handle.h"
#include "transport/hip_transport/ualink_topology.h"
#include "transport/transport.h"

namespace mooncake {

using BufferDesc = TransferMetadata::BufferDesc;

// Pool configuration constants
// These can be overridden via environment variables:
// - MC_HIP_NUM_STREAMS: number of HIP streams per device
// - MC_HIP_NUM_EVENTS: number of HIP events per device
constexpr int kDefaultNumStreams = 64;
constexpr int kDefaultNumEvents = 64;
// hipMemcpyAsync is only used for small batches below this size; large
// copies over imported fabric memory go through the copy kernel.
constexpr uint64_t kDefaultMemcpyMaxBytes = 4ull << 20;
constexpr size_t kDefaultKernelMinDesc = 4;

constexpr auto HIPX_MEM_HANDLE_TYPE_POSIX_FD =
    hipMemHandleTypePosixFileDescriptor;

struct hipxFabricHandle {
    int fd;
    int pid;
};

static_assert(sizeof(hipxFabricHandle) == hip_share::kLegacyPosixFdBytes,
              "legacy POSIX fd share handle layout changed");
static_assert(sizeof(hipIpcMemHandle_t) == hip_share::kLegacyIpcBytes,
              "legacy IPC share handle layout changed");
#ifdef MC_HIP_HAS_FABRIC_HANDLE
static_assert(sizeof(hipMemFabricHandle_t) <= hip_share::kMaxHandleBytes,
              "fabric handle does not fit the share descriptor");
#endif

// RAII wrapper for file descriptor management
struct FdGuard {
    int fd = -1;
    explicit FdGuard(int fd_val) : fd(fd_val) {}
    ~FdGuard() {
        if (fd != -1) {
            close(fd);
        }
    }
    // Disable copy/move
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
    FdGuard(FdGuard&&) = delete;
    FdGuard& operator=(FdGuard&&) = delete;
};

static bool checkHip(hipError_t result, const char* message) {
    if (result != hipSuccess) {
        LOG(ERROR) << message << " (Error code: " << result << " - "
                   << hipGetErrorString(result) << ")";
        return false;
    }
    return true;
}

static bool envFlag(const char* name) {
    const char* env = getenv(name);
    return env && env[0] != '\0' && strcmp(env, "0") != 0;
}

static uint64_t envU64(const char* name, uint64_t fallback) {
    const char* env = getenv(name);
    if (!env || !*env) return fallback;
    char* end = nullptr;
    errno = 0;
    unsigned long long value = strtoull(env, &end, 0);
    if (errno || !end || *end != '\0') {
        LOG(WARNING) << "Ignoring invalid " << name;
        return fallback;
    }
    return value;
}

// Wrapper around hipMemImportFromShareableHandle to handle ROCm version
// differences. In ROCm 7.1+, the signature was changed:
// the second argument is (void*)(uintptr_t)fd instead of a pointer to the fd
static hipError_t importFromShareableHandle(
    hipMemGenericAllocationHandle_t* handle, hipxFabricHandle* export_handle) {
    static const int runtime_version = []() {
        int version = 0;
        if (!checkHip(hipRuntimeGetVersion(&version),
                      "HipTransport: hipRuntimeGetVersion failed")) {
            return -1;
        }
        return version;
    }();

    if (runtime_version < 0) {
        return hipErrorInvalidValue;
    }

    if (runtime_version >= 70100000) {
        return hipMemImportFromShareableHandle(
            handle, (void*)(uintptr_t)export_handle->fd,
            HIPX_MEM_HANDLE_TYPE_POSIX_FD);
    } else {
        return hipMemImportFromShareableHandle(handle, export_handle,
                                               HIPX_MEM_HANDLE_TYPE_POSIX_FD);
    }
}

static int open_fd(const hipxFabricHandle& export_handle) {
    int fd = export_handle.fd;
    int pid = export_handle.pid;

    int pid_fd = (int)syscall(__NR_pidfd_open, pid, 0);
    if (pid_fd == -1) {
        LOG(ERROR) << "HIPTransport: pidfd_open error: " << strerror(errno)
                   << " ( " << pid << " " << fd << ")";
        return -1;
    }

    int open_fd = (int)syscall(__NR_pidfd_getfd, pid_fd, fd, 0);
    if (open_fd == -1) {
        LOG(ERROR) << "HIPTransport: pidfd_getfd error: " << strerror(errno)
                   << " ( " << pid << " " << fd << ")";
        close(pid_fd);
        return -1;
    }

    close(pid_fd);
    return open_fd;
}

static int openIPCHandle(const std::vector<unsigned char>& buffer,
                         void** shm_addr) {
    hipIpcMemHandle_t handle;
    memcpy(&handle, buffer.data(), sizeof(handle));
    if (!checkHip(hipIpcOpenMemHandle(shm_addr, handle,
                                      hipIpcMemLazyEnablePeerAccess),
                  "HipTransport: hipIpcOpenMemHandle failed")) {
        return -1;
    }
    return 0;
}

static int openShareableHandle(const std::vector<unsigned char>& buffer,
                               size_t length, void** shm_addr) {
    hipxFabricHandle export_handle;
    memcpy(&export_handle, buffer.data(), sizeof(export_handle));

    int opened_fd = open_fd(export_handle);
    if (opened_fd == -1) {
        LOG(ERROR) << "HIPTransport: failed to open fd";
        return -1;
    }
    // Use RAII guard for automatic fd cleanup
    FdGuard fd_guard(opened_fd);
    export_handle.fd = opened_fd;

    hipMemGenericAllocationHandle_t handle;
    if (!checkHip(importFromShareableHandle(&handle, &export_handle),
                  "HipTransport: importFromShareableHandle failed")) {
        return -1;
    }

    if (!checkHip(hipMemAddressReserve((hipDeviceptr_t*)shm_addr, length, 0,
                                       nullptr, 0),
                  "HipTransport: hipMemAddressReserve failed")) {
        (void)hipMemRelease(handle);
        return -1;
    }

    hipError_t map_err =
        hipMemMap((hipDeviceptr_t)*shm_addr, length, 0, handle, 0);
    // The mapping keeps the allocation alive.
    (void)hipMemRelease(handle);
    if (!checkHip(map_err, "HipTransport: hipMemMap failed")) {
        (void)hipMemAddressFree((hipDeviceptr_t)*shm_addr, length);
        return -1;
    }

    int device_count = 0;
    (void)hipGetDeviceCount(&device_count);
    std::vector<hipMemAccessDesc> accessDesc(device_count);
    for (int device_id = 0; device_id < device_count; ++device_id) {
        accessDesc[device_id].location.type = hipMemLocationTypeDevice;
        accessDesc[device_id].location.id = device_id;
        accessDesc[device_id].flags = hipMemAccessFlagsProtReadWrite;
    }

    if (!checkHip(hipMemSetAccess((hipDeviceptr_t)*shm_addr, length,
                                  accessDesc.data(), device_count),
                  "HipTransport: hipMemSetAccess failed")) {
        (void)hipMemUnmap((hipDeviceptr_t)*shm_addr, length);
        (void)hipMemAddressFree((hipDeviceptr_t)*shm_addr, length);
        return -1;
    }

    return 0;
}

// Device ordinal owning ptr, or -1 for host / unknown memory.
static int pointerDevice(const void* ptr) {
    if (!ptr) return -1;
    hipPointerAttribute_t attributes;
    if (hipPointerGetAttributes(&attributes, ptr) != hipSuccess) {
        // Clear the sticky error so it does not leak into caller HIP calls.
        (void)hipGetLastError();
        return -1;
    }
    return attributes.type == hipMemoryTypeDevice ? attributes.device : -1;
}

static void setupP2PAccess(int num_devices) {
    // The loop switches devices; the guard restores the caller's on return.
    HipDeviceGuard device_guard;

    auto clearStickyPeerAccessError = [](int src_device, int dst_device) {
        // hipDeviceEnablePeerAccess may leave hipErrorPeerAccessAlreadyEnabled
        // in the runtime's last-error slot. Clear it so subsequent PyTorch HIP
        // calls (for example torch.empty during the next connector init) do not
        // observe a stale error.
        hipError_t last_error = hipGetLastError();
        if (last_error != hipSuccess &&
            last_error != hipErrorPeerAccessAlreadyEnabled) {
            LOG(WARNING) << "HipTransport: unexpected sticky HIP error after "
                            "enabling P2P access from device "
                         << src_device << " to device " << dst_device << " ("
                         << hipGetErrorString(last_error) << ")";
        }
    };

    for (int i = 0; i < num_devices; ++i) {
        if (!checkHip(hipSetDevice(i), "HipTransport: failed to set device")) {
            continue;
        }

        for (int j = 0; j < num_devices; ++j) {
            if (i == j) {
                continue;
            }

            int can_access_peer = 0;
            if (!checkHip(hipDeviceCanAccessPeer(&can_access_peer, i, j),
                          "HipTransport: failed to query peer access")) {
                continue;
            }

            if (can_access_peer) {
                hipError_t result = hipDeviceEnablePeerAccess(j, 0);
                if (result == hipErrorPeerAccessAlreadyEnabled) {
                    clearStickyPeerAccessError(i, j);
                } else if (result != hipSuccess) {
                    clearStickyPeerAccessError(i, j);
                    LOG(WARNING) << "HipTransport: failed to enable P2P access "
                                    "from device "
                                 << i << " to device " << j << " ("
                                 << hipGetErrorString(result) << ")";
                }
            } else if (i < j) {
                LOG(WARNING)
                    << "HipTransport: P2P access not available between device "
                    << i << " and device " << j;
            }
        }
    }
}

static int getNumStreams() {
    const char* env = getenv("MC_HIP_NUM_STREAMS");
    if (env) {
        try {
            int value = std::stoi(env);
            if (value > 0) {
                return value;
            }
            LOG(WARNING) << "MC_HIP_NUM_STREAMS value " << value
                         << " must be positive, using default "
                         << kDefaultNumStreams;
        } catch (...) {
            LOG(WARNING) << "Invalid MC_HIP_NUM_STREAMS value, using default "
                         << kDefaultNumStreams;
        }
    }
    return kDefaultNumStreams;
}

static int getNumEvents() {
    const char* env = getenv("MC_HIP_NUM_EVENTS");
    if (env) {
        try {
            int value = std::stoi(env);
            if (value > 0) {
                return value;
            }
            LOG(WARNING) << "MC_HIP_NUM_EVENTS value " << value
                         << " must be positive, using default "
                         << kDefaultNumEvents;
        } catch (...) {
            LOG(WARNING) << "Invalid MC_HIP_NUM_EVENTS value, using default "
                         << kDefaultNumEvents;
        }
    }
    return kDefaultNumEvents;
}

static bool allDevicesSupportVmm() {
    int num_devices = 0;
    if (!checkHip(hipGetDeviceCount(&num_devices),
                  "HipTransport: hipGetDeviceCount failed")) {
        return false;
    }

    if (num_devices == 0) {
        LOG(ERROR) << "HipTransport: no device found";
        return false;
    }

    for (int device_id = 0; device_id < num_devices; ++device_id) {
        int vmm_supported = 0;
        hipError_t result = hipDeviceGetAttribute(
            &vmm_supported, hipDeviceAttributeVirtualMemoryManagementSupported,
            device_id);
        if (result != hipSuccess || !vmm_supported) {
            LOG(WARNING) << "HipTransport: Device " << device_id
                         << " does not support virtual memory management";
            return false;
        }
    }
    return true;
}

// Legacy same-host sharing of VMM memory through POSIX fds. Enabled only when
// MC_USE_HIP_IPC=0 or MC_USE_NVLINK_IPC=0 is explicitly set.
static bool legacyPosixFdMode() {
    const char* hip_ipc = getenv("MC_USE_HIP_IPC");
    const char* nvlink_ipc = getenv("MC_USE_NVLINK_IPC");

    bool enabled = (hip_ipc && strcmp(hip_ipc, "0") == 0) ||
                   (nvlink_ipc && strcmp(nvlink_ipc, "0") == 0);
    return enabled && allDevicesSupportVmm();
}

#ifdef MC_HIP_HAS_FABRIC_HANDLE
// The attribute only says the GPU has UALink; an accelerator that is not
// provisioned yet still creates fabric allocations and fails at export. Every
// step is expected to fail off-fabric, so each sticky error is consumed here
// rather than surfacing at an unrelated call later.
static bool fabricRoundTrip(int device) {
    int supported = 0;
    if (hipDeviceGetAttribute(&supported,
                              hipDeviceAttributeHandleTypeFabricSupported,
                              device) != hipSuccess ||
        !supported) {
        (void)hipGetLastError();
        return false;
    }
    // hipMemImportFromShareableHandle imports onto the current device.
    HipDeviceGuard guard(device);
    if (!guard.set_ok()) {
        (void)hipGetLastError();
        return false;
    }
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    prop.requestedHandleTypes = hipMemHandleTypeFabric;
    size_t granularity = 0;
    if (hipMemGetAllocationGranularity(&granularity, &prop,
                                       hipMemAllocationGranularityMinimum) !=
            hipSuccess ||
        granularity == 0) {
        (void)hipGetLastError();
        return false;
    }
    hipMemGenericAllocationHandle_t handle;
    if (hipMemCreate(&handle, granularity, &prop, 0) != hipSuccess) {
        (void)hipGetLastError();
        return false;
    }
    hipMemFabricHandle_t shared = {};
    hipMemGenericAllocationHandle_t imported;
    const bool ok =
        hipMemExportToShareableHandle(&shared, handle, hipMemHandleTypeFabric,
                                      0) == hipSuccess &&
        hipMemImportFromShareableHandle(&imported, &shared,
                                        hipMemHandleTypeFabric) == hipSuccess;
    if (ok)
        (void)hipMemRelease(imported);
    else
        (void)hipGetLastError();
    (void)hipMemRelease(handle);
    if (!ok) {
        LOG(WARNING) << "HipTransport: device " << device
                     << " reports UALink but cannot export or import a fabric "
                        "handle (accelerator not provisioned?); fabric "
                        "export is disabled on it";
    }
    return ok;
}
#endif

static bool deviceSupportsFabric(int device) {
#ifdef MC_HIP_HAS_FABRIC_HANDLE
    static std::mutex mutex;
    static std::map<int, bool> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(device);
    if (it != cache.end()) return it->second;
    const bool supported = fabricRoundTrip(device);
    cache[device] = supported;
    return supported;
#else
    (void)device;
    return false;
#endif
}

static bool fabricExportEnabled() {
#ifdef MC_HIP_HAS_FABRIC_HANDLE
    return !ualink::fabricDisabledByEnv();
#else
    return false;
#endif
}

static const std::vector<ualink::VpodInfo>& localVpods() {
    static const std::vector<ualink::VpodInfo> vpods = []() {
        std::vector<ualink::VpodInfo> result;
        int count = 0;
        if (hipGetDeviceCount(&count) != hipSuccess) {
            (void)hipGetLastError();
            return result;
        }
        for (int device = 0; device < count; ++device) {
            char bdf[64] = {0};
            if (hipDeviceGetPCIBusId(bdf, sizeof(bdf), device) != hipSuccess) {
                (void)hipGetLastError();
                result.emplace_back();
                continue;
            }
            result.push_back(ualink::resolveVpod(bdf));
        }
        return result;
    }();
    return vpods;
}

static ualink::VpodInfo localVpod(int device) {
    const auto& vpods = localVpods();
    if (device < 0 || static_cast<size_t>(device) >= vpods.size()) return {};
    return vpods[device];
}

static size_t fabricMapAlignment() {
#ifdef MC_HIP_HAS_FABRIC_HANDLE
    static const size_t alignment = []() -> size_t {
        hipMemAllocationProp prop = {};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = 0;
        prop.requestedHandleTypes = hipMemHandleTypeFabric;
        size_t granularity = 0;
        if (hipMemGetAllocationGranularity(
                &granularity, &prop, hipMemAllocationGranularityRecommended) !=
            hipSuccess) {
            (void)hipGetLastError();
            return 0;
        }
        return granularity;
    }();
    return alignment;
#else
    return 0;
#endif
}

// Maps every chunk of a fabric export into one contiguous VA range.
static int importFabric(const hip_share::Descriptor& share, uint64_t length,
                        void** base_out) {
#ifdef MC_HIP_HAS_FABRIC_HANDLE
    uint64_t total = 0;
    for (const auto& chunk : share.chunks) total += chunk.size;
    if (total != length) {
        LOG(ERROR) << "HipTransport: fabric chunks cover " << total
                   << " bytes but buffer length is " << length;
        return ERR_INVALID_ARGUMENT;
    }
    void* base = nullptr;
    if (!checkHip(hipMemAddressReserve((hipDeviceptr_t*)&base, length,
                                       fabricMapAlignment(), nullptr, 0),
                  "HipTransport: hipMemAddressReserve failed")) {
        return ERR_MEMORY;
    }
    size_t mapped = 0;
    auto rollback = [&]() {
        for (size_t i = 0; i < mapped; ++i) {
            const auto& chunk = share.chunks[i];
            (void)hipMemUnmap((hipDeviceptr_t)((char*)base + chunk.offset),
                              chunk.size);
        }
        (void)hipMemAddressFree((hipDeviceptr_t)base, length);
    };
    for (const auto& chunk : share.chunks) {
        hipMemFabricHandle_t fabric_handle = {};
        memcpy(fabric_handle.data, chunk.handle.data(),
               std::min(chunk.handle.size(), sizeof(fabric_handle.data)));
        hipMemGenericAllocationHandle_t handle;
        if (!checkHip(hipMemImportFromShareableHandle(&handle, &fabric_handle,
                                                      hipMemHandleTypeFabric),
                      "HipTransport: fabric handle import failed")) {
            rollback();
            return ERR_MEMORY;
        }
        hipError_t err = hipMemMap((hipDeviceptr_t)((char*)base + chunk.offset),
                                   chunk.size, 0, handle, 0);
        (void)hipMemRelease(handle);
        if (!checkHip(err, "HipTransport: hipMemMap of fabric chunk failed")) {
            rollback();
            return ERR_MEMORY;
        }
        ++mapped;
    }
    *base_out = base;
    return 0;
#else
    (void)share;
    (void)length;
    (void)base_out;
    LOG(ERROR) << "HipTransport: built without HIP fabric handle support";
    return ERR_NOT_IMPLEMENTED;
#endif
}

// Exports [addr, addr + length) as fabric handles, one per backing
// allocation. The published range is widened to whole allocations.
static int exportFabric(void* addr, size_t length, int device,
                        BufferDesc& desc) {
#ifdef MC_HIP_HAS_FABRIC_HANDLE
    hip_share::Descriptor share;
    share.kind = hip_share::Kind::kFabric;
    share.owner_pid = getpid();
    share.owner_device = device;
    auto vpod = localVpod(device);
    if (vpod.valid) {
        share.ppod_id = vpod.ppod_id;
        share.vpod_id = vpod.vpod_id;
    }

    const uint64_t end = (uint64_t)addr + length;
    uint64_t cursor = (uint64_t)addr;
    uint64_t region_base = 0;
    while (cursor < end) {
        hipMemGenericAllocationHandle_t handle;
        if (hipMemRetainAllocationHandle(&handle, (void*)cursor) !=
            hipSuccess) {
            (void)hipGetLastError();
            LOG(WARNING) << "HipTransport: " << (void*)cursor
                         << " is not backed by a VMM allocation";
            return ERR_INVALID_ARGUMENT;
        }
        hipDeviceptr_t base = nullptr;
        size_t size = 0;
        hipError_t err =
            hipMemGetAddressRange(&base, &size, (hipDeviceptr_t)cursor);
        hipMemFabricHandle_t fabric_handle = {};
        if (err == hipSuccess)
            err = hipMemExportToShareableHandle(&fabric_handle, handle,
                                                hipMemHandleTypeFabric, 0);
        (void)hipMemRelease(handle);
        if (err != hipSuccess || size == 0) {
            (void)hipGetLastError();
            LOG(WARNING) << "HipTransport: fabric export at " << (void*)cursor
                         << " failed: " << hipGetErrorString(err);
            return ERR_INVALID_ARGUMENT;
        }
        if (share.chunks.empty()) region_base = (uint64_t)base;
        hip_share::Chunk chunk;
        chunk.offset = (uint64_t)base - region_base;
        chunk.size = size;
        chunk.handle.assign(fabric_handle.data,
                            fabric_handle.data + sizeof(fabric_handle.data));
        if (chunk.offset !=
            (share.chunks.empty()
                 ? 0
                 : share.chunks.back().offset + share.chunks.back().size)) {
            LOG(WARNING) << "HipTransport: allocations under " << addr
                         << " are not virtually contiguous";
            return ERR_INVALID_ARGUMENT;
        }
        share.chunks.push_back(std::move(chunk));
        cursor = (uint64_t)base + size;
    }

    desc.addr = region_base;
    desc.length = cursor - region_base;
    desc.shm_name = hip_share::encodeToShmName(share);
    if (globalConfig().trace) {
        LOG(INFO) << "HipTransport: fabric export " << addr << " +" << length
                  << " -> base " << (void*)region_base << " +" << desc.length
                  << " in " << share.chunks.size() << " chunk(s), vpod "
                  << (vpod.valid ? std::to_string(vpod.vpod_id) : "none");
    }
    return 0;
#else
    (void)addr;
    (void)length;
    (void)device;
    (void)desc;
    return ERR_NOT_IMPLEMENTED;
#endif
}

struct HipTransport::RemoteMapping {
    hip_share::Kind kind = hip_share::Kind::kUnknown;
    void* base = nullptr;
    uint64_t length = 0;
    std::string shm_name;
    // Devices granted access through hipMemSetAccess (fabric only).
    uint64_t access_mask = 0;
    bool all_access = false;

    bool hasAccess(int device) const {
        return all_access ||
               (device >= 0 && device < 64 && (access_mask & (1ull << device)));
    }

    ~RemoteMapping() {
        if (!base) return;
        if (kind == hip_share::Kind::kIpc) {
            (void)hipIpcCloseMemHandle(base);
        } else {
            (void)hipMemUnmap((hipDeviceptr_t)base, length);
            (void)hipMemAddressFree((hipDeviceptr_t)base, length);
        }
    }
};

HipTransport::HipTransport()
    : use_fabric_mem_(legacyPosixFdMode()),
      fabric_export_(fabricExportEnabled()),
      access_all_devices_(envFlag("MC_UALOE_ACCESS_ALL_DEVICES")),
      memcpy_max_bytes_(
          envU64("MC_HIP_MEMCPY_MAX_BYTES", kDefaultMemcpyMaxBytes)),
      kernel_min_desc_(
          envU64("MC_UALOE_KERNEL_MIN_DESC", kDefaultKernelMinDesc)),
      stream_pool_(getNumStreams()),
      event_pool_(getNumEvents()) {
    if (!checkHip(hipGetDeviceCount(&num_devices_),
                  "HipTransport: hipGetDeviceCount failed")) {
        num_devices_ = 0;
    }
    const char* kernel_env = getenv("MC_HIP_COPY_KERNEL");
    if (kernel_env && strcmp(kernel_env, "0") == 0) {
        kernel_min_desc_ = SIZE_MAX;
        memcpy_max_bytes_ = UINT64_MAX;
    }

    // Enable P2P access for IPC mode
    if (!use_fabric_mem_) setupP2PAccess(num_devices_);

    auto vpod = localVpod(0);
    LOG(INFO) << "HipTransport: devices=" << num_devices_
              << " posix_fd=" << use_fabric_mem_ << " fabric_export="
              << (fabric_export_ && deviceSupportsFabric(0)) << " vpod="
              << (vpod.valid ? std::to_string(vpod.vpod_id) + "@" +
                                   (vpod.ppod_id.empty() ? "*" : vpod.ppod_id) +
                                   " (" + vpod.source + ")"
                             : std::string("none"));
}

HipTransport::~HipTransport() {
    RWSpinlock::WriteGuard guard(remap_lock_);
    remap_entries_.clear();
}

int HipTransport::install(std::string& local_server_name,
                          std::shared_ptr<TransferMetadata> metadata,
                          std::shared_ptr<Topology> topology) {
    metadata_ = metadata;
    local_server_name_ = local_server_name;

    // Compose with any local segment another transport (e.g. RDMA) already
    // installed instead of overwriting it, so a single-node segment can
    // advertise both protocols (e.g. "rdma,hip"). Work on a copy (the map may
    // hand back a descriptor other threads are reading) and publish the new
    // one atomically via addLocalSegment.
    auto old_desc = metadata_->getSegmentDescByID(LOCAL_SEGMENT_ID);
    auto desc = std::make_shared<SegmentDesc>();
    if (!desc) return ERR_MEMORY;
    if (old_desc) *desc = *old_desc;

    desc->name = local_server_name_;
#ifdef ENABLE_MULTI_PROTOCOL
    if (desc->protocol.empty()) {
        desc->protocol = "hip";
    } else if (desc->protocol.find("hip") == std::string::npos) {
        desc->protocol += ",hip";
    }
#else
    desc->protocol = "hip";
#endif

    metadata_->addLocalSegment(LOCAL_SEGMENT_ID, local_server_name_,
                               std::move(desc));
    return 0;
}

int HipTransport::sourceDevice(const void* source, uint64_t length,
                               PrepareCache& cache) {
    const uint64_t begin = (uint64_t)source;
    if (begin >= cache.src_begin && begin + length <= cache.src_end)
        return cache.src_device;
    {
        RWSpinlock::ReadGuard guard(local_regions_lock_);
        auto it = local_regions_.upper_bound(begin);
        if (it != local_regions_.begin()) {
            --it;
            if (begin + length <= it->second.first) {
                cache.src_begin = it->first;
                cache.src_end = it->second.first;
                cache.src_device = it->second.second;
                return cache.src_device;
            }
        }
    }
    return pointerDevice(source);
}

Status HipTransport::prepareRequest(const TransferRequest& request,
                                    TransferTask& task, Prepared& prepared,
                                    PrepareCache& cache) {
    int device_id = sourceDevice(request.source, request.length, cache);
    const bool src_on_device = device_id >= 0;
    if (!src_on_device) {
        if (hipGetDevice(&device_id) != hipSuccess) device_id = 0;
    }
    if (device_id != cache.current_device) {
        if (!checkHip(hipSetDevice(device_id),
                      "HipTransport: failed to set device context")) {
            return Status::InvalidArgument("Failed to set device context");
        }
        cache.current_device = device_id;
    }

    uint64_t dest_addr = request.target_offset;
    bool dst_on_device = true;
    if (request.target_id != LOCAL_SEGMENT_ID) {
        const bool hit =
            cache.mapping && cache.target_id == request.target_id &&
            cache.buf_device == device_id && dest_addr >= cache.buf_addr &&
            dest_addr + request.length <= cache.buf_addr + cache.buf_length;
        if (!hit) {
            if (cache.target_id != request.target_id || !cache.desc) {
                cache.desc = metadata_->getSegmentDescByID(request.target_id);
                cache.target_id = request.target_id;
            }
            cache.mapping.reset();
            if (!cache.desc) {
                LOG(ERROR) << "HipTransport: unknown segment "
                           << request.target_id;
                return Status::Memory("device memory not registered");
            }
            for (const auto& entry : cache.desc->buffers) {
#ifdef ENABLE_MULTI_PROTOCOL
                if (!entry.protocol.empty() && entry.protocol != "hip")
                    continue;
#endif
                if (entry.shm_name.empty() || entry.addr > dest_addr ||
                    dest_addr + request.length > entry.addr + entry.length)
                    continue;
                int rc = mapBuffer(request.target_id, entry, device_id,
                                   cache.mapping);
                if (rc) return Status::Memory("failed to map remote buffer");
                cache.buf_addr = entry.addr;
                cache.buf_length = entry.length;
                cache.buf_device = device_id;
                break;
            }
            if (!cache.mapping) {
                LOG(ERROR) << "Requested address " << (void*)dest_addr << " to "
                           << (void*)(dest_addr + request.length)
                           << " not found!";
                return Status::Memory("device memory not registered");
            }
        }
        prepared.mapping = cache.mapping;
        dest_addr = dest_addr - cache.buf_addr + (uint64_t)cache.mapping->base;
    } else {
        dst_on_device = pointerDevice((void*)dest_addr) >= 0;
    }

    task.total_bytes = request.length;

    Slice* slice = getSliceCache().allocate();
    slice->source_addr = (char*)request.source;
    slice->local.dest_addr = (char*)dest_addr;
    slice->length = request.length;
    slice->opcode = request.opcode;
    slice->task = &task;
    slice->target_id = request.target_id;
    slice->status = Slice::PENDING;
    task.slice_list.push_back(slice);
    __sync_fetch_and_add(&task.slice_count, 1);

    prepared.slice = slice;
    prepared.device_id = device_id;
    if (request.opcode == TransferRequest::READ) {
        prepared.dst = request.source;
        prepared.src = (const void*)dest_addr;
    } else {
        prepared.dst = (void*)dest_addr;
        prepared.src = request.source;
    }
    // The kernel dereferences both sides; pageable host memory must go
    // through hipMemcpyAsync.
    prepared.kernel_eligible = src_on_device && dst_on_device;
    return Status::OK();
}

Status HipTransport::submitRequests(
    const std::vector<std::pair<const TransferRequest*, TransferTask*>>& work) {
    HipDeviceGuard device_guard;

    std::vector<Prepared> prepared(work.size());
    PrepareCache cache;
    for (size_t i = 0; i < work.size(); ++i) {
        Status status =
            prepareRequest(*work[i].first, *work[i].second, prepared[i], cache);
        if (!status.ok()) {
            // Nothing has been launched yet; fail what was prepared so the
            // tasks still complete.
            for (size_t j = 0; j < i; ++j) prepared[j].slice->markFailed();
            return status;
        }
    }

    // Per device: one kernel launch for the whole group, unless the group is
    // small enough that per-request hipMemcpyAsync is cheaper.
    std::map<int, std::vector<size_t>> by_device;
    std::vector<size_t> memcpy_items;
    for (size_t i = 0; i < prepared.size(); ++i) {
        if (prepared[i].kernel_eligible)
            by_device[prepared[i].device_id].push_back(i);
        else
            memcpy_items.push_back(i);
    }

    struct KernelGroup {
        std::unique_ptr<HipBatchCopier::Launch> launch;
        std::vector<size_t> items;
    };
    std::vector<KernelGroup> kernel_groups;
    auto& copier = HipBatchCopier::instance();
    for (auto& [device, items] : by_device) {
        bool use_kernel = items.size() >= kernel_min_desc_;
        for (size_t i = 0; !use_kernel && i < items.size(); ++i)
            use_kernel = prepared[items[i]].slice->length >= memcpy_max_bytes_;
        if (use_kernel && copier.ready(device)) {
            std::vector<HipBatchCopier::Item> copies;
            copies.reserve(items.size());
            for (size_t idx : items)
                copies.push_back({prepared[idx].dst, prepared[idx].src,
                                  prepared[idx].slice->length});
            KernelGroup group;
            (void)hipSetDevice(device);
            hipError_t err = copier.submit(device, copies, group.launch);
            if (err == hipSuccess) {
                group.items = std::move(items);
                kernel_groups.push_back(std::move(group));
                continue;
            }
            (void)hipGetLastError();
            LOG(WARNING) << "HipTransport: copy kernel launch failed on device "
                         << device << " (" << hipGetErrorString(err)
                         << "), using hipMemcpyAsync";
        }
        memcpy_items.insert(memcpy_items.end(), items.begin(), items.end());
    }

    struct PendingCopy {
        hipEvent_t event;
        int device_id;
        size_t item;
    };
    std::vector<PendingCopy> pending;
    pending.reserve(memcpy_items.size());
    for (size_t idx : memcpy_items) {
        auto& p = prepared[idx];
        (void)hipSetDevice(p.device_id);
        hipEvent_t event = event_pool_.getEvent(p.device_id);
        hipStream_t stream = stream_pool_.getNextStream(p.device_id);
        if (event == nullptr || stream == nullptr) {
            if (event != nullptr) event_pool_.putEvent(event, p.device_id);
            LOG(ERROR) << "HipTransport: failed to get event or stream";
            p.slice->markFailed();
            continue;
        }
        hipError_t err = hipMemcpyAsync(p.dst, p.src, p.slice->length,
                                        hipMemcpyDefault, stream);
        if (err == hipSuccess) err = hipEventRecord(event, stream);
        if (!checkHip(err, "HipTransport: async copy failed")) {
            event_pool_.putEvent(event, p.device_id);
            p.slice->markFailed();
            continue;
        }
        pending.push_back({event, p.device_id, idx});
    }

    for (auto& group : kernel_groups) {
        hipError_t err = copier.wait(group.launch);
        if (err != hipSuccess)
            LOG(ERROR) << "HipTransport: copy kernel failed: "
                       << hipGetErrorString(err);
        for (size_t idx : group.items) {
            if (err == hipSuccess)
                prepared[idx].slice->markSuccess();
            else
                prepared[idx].slice->markFailed();
        }
    }
    for (auto& pc : pending) {
        hipError_t err = hipEventSynchronize(pc.event);
        if (err == hipSuccess) {
            prepared[pc.item].slice->markSuccess();
        } else {
            LOG(ERROR) << "HipTransport: hipEventSynchronize failed: "
                       << hipGetErrorString(err);
            prepared[pc.item].slice->markFailed();
        }
        event_pool_.putEvent(pc.event, pc.device_id);
    }
    return Status::OK();
}

Status HipTransport::submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest>& entries) {
    auto& batch_desc = *((BatchDesc*)(batch_id));
    if (batch_desc.task_list.size() + entries.size() > batch_desc.batch_size) {
        LOG(ERROR) << "HipTransport: Exceed the limitation of current batch's "
                      "capacity";
        return Status::InvalidArgument(
            "HipTransport: Exceed the limitation of capacity, batch id: " +
            std::to_string(batch_id));
    }

    size_t task_id = batch_desc.task_list.size();
    batch_desc.task_list.resize(task_id + entries.size());

    std::vector<std::pair<const TransferRequest*, TransferTask*>> work;
    work.reserve(entries.size());
    for (auto& request : entries)
        work.emplace_back(&request, &batch_desc.task_list[task_id++]);
    return submitRequests(work);
}

Status HipTransport::getTransferStatus(BatchID batch_id, size_t task_id,
                                       TransferStatus& status) {
    auto& batch_desc = *((BatchDesc*)(batch_id));
    const size_t task_count = batch_desc.task_list.size();

    if (task_id >= task_count) {
        return Status::InvalidArgument(
            "HipTransport::getTransferStatus invalid argument, batch id: " +
            std::to_string(batch_id));
    }

    // Get task and status info
    auto& task = batch_desc.task_list[task_id];
    status.transferred_bytes = task.transferred_bytes;
    uint64_t success_slice_count = task.success_slice_count;
    uint64_t failed_slice_count = task.failed_slice_count;

    // Determine completion status
    if (success_slice_count + failed_slice_count == task.slice_count) {
        if (failed_slice_count) {
            status.s = TransferStatusEnum::FAILED;
        } else {
            status.s = TransferStatusEnum::COMPLETED;
        }
        task.is_finished = true;
    } else {
        status.s = TransferStatusEnum::WAITING;
    }

    return Status::OK();
}

Status HipTransport::submitTransferTask(
    const std::vector<TransferTask*>& task_list) {
    std::vector<std::pair<const TransferRequest*, TransferTask*>> work;
    work.reserve(task_list.size());
    for (auto* task_ptr : task_list) {
        assert(task_ptr);
        assert(task_ptr->request);
        work.emplace_back(task_ptr->request, task_ptr);
    }
    return submitRequests(work);
}

int HipTransport::exportBuffer(void* addr, size_t length, int device,
                               BufferDesc& desc) {
    hipMemGenericAllocationHandle_t handle;
    const bool vmm = hipMemRetainAllocationHandle(&handle, addr) == hipSuccess;
    if (vmm)
        (void)hipMemRelease(handle);
    else
        (void)hipGetLastError();

    if (vmm && fabric_export_ && deviceSupportsFabric(device)) {
        if (exportFabric(addr, length, device, desc) == 0) return 0;
        LOG(WARNING) << "HipTransport: " << addr
                     << " is not fabric-exportable, falling back to "
                     << (use_fabric_mem_ ? "POSIX fd" : "IPC");
    }

    if (use_fabric_mem_) {
        if (!vmm) {
            LOG(WARNING) << "Memory region " << addr
                         << " is not allocated by hipMemCreate, "
                         << "but it can be used as local buffer";
            return 1;
        }
        if (!checkHip(hipMemRetainAllocationHandle(&handle, addr),
                      "HipTransport: hipMemRetainAllocationHandle failed")) {
            return -1;
        }
        void* real_addr;
        size_t real_size;
        hipError_t result = hipMemGetAddressRange(
            (hipDeviceptr_t*)&real_addr, &real_size, (hipDeviceptr_t)addr);
        if (result != hipSuccess) {
            LOG(WARNING) << "HipTransport: hipMemGetAddressRange failed: "
                         << result;
            const uint64_t granularity = 2ULL * 1024 * 1024;
            real_addr = addr;
            real_size = (length + granularity - 1) & ~(granularity - 1);
        }
        hipxFabricHandle export_handle_raw = {};
        result = hipMemExportToShareableHandle(
            &export_handle_raw, handle, HIPX_MEM_HANDLE_TYPE_POSIX_FD, 0);
        (void)hipMemRelease(handle);
        if (!checkHip(result,
                      "HipTransport: hipMemExportToShareableHandle failed")) {
            return -1;
        }
        export_handle_raw.pid = getpid();
        desc.addr = (uint64_t)real_addr;
        desc.length = real_size;
        desc.shm_name = serializeBinaryData((const void*)&export_handle_raw,
                                            sizeof(hipxFabricHandle));
        return 0;
    }

    hipIpcMemHandle_t ipc_handle;
    hipError_t ipc_err = hipIpcGetMemHandle(&ipc_handle, addr);
    if (ipc_err != hipSuccess && vmm) {
        // VMM memory cannot be shared through IPC. Leave it to RDMA/TCP
        // rather than failing the registration on every transport.
        (void)hipGetLastError();
        LOG(WARNING) << "HipTransport: VMM buffer " << addr
                     << " has no usable share handle (fabric "
                     << (fabric_export_ ? "export failed" : "disabled")
                     << "), not exposing it through hip";
        return 1;
    }
    if (!checkHip(ipc_err, "HipTransport: hipIpcGetMemHandle failed")) {
        return -1;
    }
    desc.addr = (uint64_t)addr;
    desc.length = length;
    desc.shm_name = serializeBinaryData(&ipc_handle, sizeof(hipIpcMemHandle_t));
    return 0;
}

int HipTransport::registerLocalMemory(void* addr, size_t length,
                                      const std::string& location,
                                      bool remote_accessible,
                                      bool update_metadata) {
    std::lock_guard<std::mutex> lock(register_mutex_);

    if (globalConfig().trace) {
        LOG(INFO) << "register memory: addr " << addr << ", length " << length;
    }

    // Only device memory can be shared. Callers that register a batch across
    // all transports (e.g. SGLang's PD metadata/aux buffers, which live in
    // host memory) also hand those host buffers to this transport. Skip
    // non-device memory gracefully and let RDMA/TCP register it; returning an
    // error would roll back the entire multi-protocol batch and tear down
    // every session.
    const int device = pointerDevice(addr);
    if (device < 0) {
        if (globalConfig().trace) {
            LOG(INFO) << "HipTransport: skipping non-device memory " << addr
                      << ", leaving it to other transports";
        }
        return 0;
    }

    (void)remote_accessible;
    BufferDesc desc;
    int rc = exportBuffer(addr, length, device, desc);
    if (rc == 1) return 0;
    if (rc) return rc;
    desc.name = location;
#ifdef ENABLE_MULTI_PROTOCOL
    desc.protocol = "hip";
#endif
    published_base_[(uint64_t)addr] = desc.addr;
    {
        RWSpinlock::WriteGuard guard(local_regions_lock_);
        local_regions_[(uint64_t)addr] = {(uint64_t)addr + length, device};
    }
    return metadata_->addLocalMemoryBuffer(desc, update_metadata);
}

int HipTransport::unregisterLocalMemory(void* addr, bool update_metadata) {
    std::lock_guard<std::mutex> lock(register_mutex_);
    auto it = published_base_.find((uint64_t)addr);
    if (it == published_base_.end()) return 0;
    uint64_t base = it->second;
    published_base_.erase(it);
    {
        RWSpinlock::WriteGuard guard(local_regions_lock_);
        local_regions_.erase((uint64_t)addr);
    }
    return metadata_->removeLocalMemoryBuffer((void*)base, update_metadata);
}

int HipTransport::openMapping(const BufferDesc& entry, int local_device,
                              MappingPtr& mapping) {
    hip_share::Descriptor share;
    if (!hip_share::decodeFromShmName(entry.shm_name, share) ||
        share.chunks.empty()) {
        LOG(ERROR) << "HipTransport: unrecognized share handle for buffer "
                   << (void*)entry.addr;
        return ERR_INVALID_ARGUMENT;
    }

    auto result = std::make_shared<RemoteMapping>();
    result->kind = share.kind;
    result->length = entry.length;
    result->shm_name = entry.shm_name;
    const auto start = std::chrono::steady_clock::now();
    int rc = -1;
    switch (share.kind) {
        case hip_share::Kind::kIpc: {
            std::vector<unsigned char> raw(share.chunks[0].handle.begin(),
                                           share.chunks[0].handle.end());
            rc = openIPCHandle(raw, &result->base);
            result->all_access = true;
            break;
        }
        case hip_share::Kind::kPosixFd: {
            std::vector<unsigned char> raw(share.chunks[0].handle.begin(),
                                           share.chunks[0].handle.end());
            rc = openShareableHandle(raw, entry.length, &result->base);
            result->all_access = true;
            break;
        }
        case hip_share::Kind::kFabric:
            rc = importFabric(share, entry.length, &result->base);
            break;
        default:
            break;
    }
    if (rc != 0) {
        result->base = nullptr;
        LOG(ERROR) << "HipTransport: failed to open "
                   << hip_share::kindName(share.kind) << " buffer "
                   << (void*)entry.addr << " +" << entry.length;
        return rc < 0 ? rc : -1;
    }
    if (globalConfig().trace) {
        LOG(INFO) << "HipTransport: opened " << hip_share::kindName(share.kind)
                  << " buffer " << (void*)entry.addr << " +" << entry.length
                  << " (" << share.chunks.size() << " chunk(s)) at "
                  << result->base << " for device " << local_device << " in "
                  << std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - start)
                         .count()
                  << " ms";
    }
    mapping = std::move(result);
    return 0;
}

int HipTransport::grantAccess(RemoteMapping& mapping, int local_device) {
    if (mapping.hasAccess(local_device)) return 0;
    std::vector<hipMemAccessDesc> access;
    auto add = [&](int device) {
        hipMemAccessDesc desc = {};
        desc.location.type = hipMemLocationTypeDevice;
        desc.location.id = device;
        desc.flags = hipMemAccessFlagsProtReadWrite;
        access.push_back(desc);
    };
    if (access_all_devices_) {
        for (int device = 0; device < num_devices_; ++device) add(device);
    } else {
        add(local_device);
    }
    if (!checkHip(hipMemSetAccess((hipDeviceptr_t)mapping.base, mapping.length,
                                  access.data(), access.size()),
                  "HipTransport: hipMemSetAccess on imported buffer failed")) {
        return ERR_MEMORY;
    }
    if (access_all_devices_)
        mapping.all_access = true;
    else if (local_device >= 0 && local_device < 64)
        mapping.access_mask |= 1ull << local_device;
    return 0;
}

int HipTransport::mapBuffer(uint64_t target_id, const BufferDesc& entry,
                            int local_device, MappingPtr& mapping) {
    const auto key = std::make_pair(target_id, entry.addr);
    {
        RWSpinlock::ReadGuard guard(remap_lock_);
        auto it = remap_entries_.find(key);
        if (it != remap_entries_.end() &&
            it->second->shm_name == entry.shm_name &&
            it->second->hasAccess(local_device)) {
            mapping = it->second;
            return 0;
        }
    }

    RWSpinlock::WriteGuard guard(remap_lock_);
    auto& slot = remap_entries_[key];
    // A changed handle means the peer re-registered the range; in-flight
    // users keep the old mapping alive through their shared_ptr.
    if (slot && slot->shm_name != entry.shm_name) slot.reset();
    if (!slot) {
        MappingPtr opened;
        int rc = openMapping(entry, local_device, opened);
        if (rc) {
            remap_entries_.erase(key);
            return rc;
        }
        slot = std::move(opened);
    }
    int rc = grantAccess(*slot, local_device);
    if (rc) return rc;
    mapping = slot;
    return 0;
}

int HipTransport::relocate(uint64_t& dest_addr, uint64_t length,
                           uint64_t target_id, int local_device,
                           MappingPtr& mapping) {
    auto desc = metadata_->getSegmentDescByID(target_id);
    if (!desc) {
        LOG(ERROR) << "HipTransport: unknown segment " << target_id;
        return ERR_INVALID_ARGUMENT;
    }

    for (auto& entry : desc->buffers) {
#ifdef ENABLE_MULTI_PROTOCOL
        if (!entry.protocol.empty() && entry.protocol != "hip") continue;
#endif
        if (entry.shm_name.empty() || entry.addr > dest_addr ||
            dest_addr + length > entry.addr + entry.length)
            continue;
        int rc = mapBuffer(target_id, entry, local_device, mapping);
        if (rc) return rc;
        dest_addr = dest_addr - entry.addr + (uint64_t)mapping->base;
        return 0;
    }
    LOG(ERROR) << "Requested address " << (void*)dest_addr << " to "
               << (void*)(dest_addr + length) << " not found!";
    return ERR_INVALID_ARGUMENT;
}

int HipTransport::relocateSharedMemoryAddress(uint64_t& dest_addr,
                                              uint64_t length,
                                              uint64_t target_id) {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) device = 0;
    MappingPtr mapping;
    return relocate(dest_addr, length, target_id, device, mapping);
}

int HipTransport::bindSegment(SegmentID segment_id, int local_device) {
    auto desc = metadata_->getSegmentDescByID(segment_id);
    if (!desc) {
        LOG(ERROR) << "HipTransport: bindSegment on unknown segment "
                   << segment_id;
        return ERR_INVALID_ARGUMENT;
    }
    std::vector<int> devices;
    if (local_device >= 0) {
        devices.push_back(local_device);
    } else {
        for (int device = 0; device < num_devices_; ++device)
            devices.push_back(device);
    }

    HipDeviceGuard device_guard;
    const auto start = std::chrono::steady_clock::now();
    size_t bound = 0;
    for (const auto& entry : desc->buffers) {
#ifdef ENABLE_MULTI_PROTOCOL
        if (!entry.protocol.empty() && entry.protocol != "hip") continue;
#endif
        if (entry.shm_name.empty()) continue;
        for (int device : devices) {
            if (!checkHip(hipSetDevice(device),
                          "HipTransport: bindSegment hipSetDevice failed"))
                return ERR_DEVICE_NOT_FOUND;
            MappingPtr mapping;
            int rc = mapBuffer(segment_id, entry, device, mapping);
            if (rc) {
                LOG(ERROR) << "HipTransport: bindSegment " << desc->name
                           << " failed on buffer " << (void*)entry.addr << " +"
                           << entry.length << " for device " << device;
                return rc;
            }
        }
        ++bound;
    }
    LOG(INFO) << "HipTransport: bound " << bound << " buffer(s) of segment "
              << desc->name << " for " << devices.size() << " device(s) in "
              << std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << " ms";
    return 0;
}

void HipTransport::invalidateSegment(SegmentID segment_id) {
    RWSpinlock::WriteGuard guard(remap_lock_);
    for (auto it = remap_entries_.begin(); it != remap_entries_.end();) {
        if (it->first.first == segment_id)
            it = remap_entries_.erase(it);
        else
            ++it;
    }
}

bool HipTransport::isFabricReachable(const std::string& shm_name) {
    if (ualink::fabricDisabledByEnv()) return false;
    static std::mutex mutex;
    static std::unordered_map<std::string, bool> cache;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = cache.find(shm_name);
        if (it != cache.end()) return it->second;
    }
    hip_share::Descriptor share;
    bool reachable = hip_share::decodeFromShmName(shm_name, share) &&
                     share.kind == hip_share::Kind::kFabric;
    if (reachable) {
        reachable = false;
        for (const auto& vpod : localVpods()) {
            if (ualink::sameVpod(vpod, share.ppod_id, share.vpod_id)) {
                reachable = true;
                break;
            }
        }
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (cache.size() >= 4096) cache.clear();
    cache[shm_name] = reachable;
    return reachable;
}

int HipTransport::registerLocalMemoryBatch(
    const std::vector<Transport::BufferEntry>& buffer_list,
    const std::string& location) {
    for (auto& buffer : buffer_list) {
        int rc = registerLocalMemory(buffer.addr, buffer.length, location, true,
                                     false);
        if (rc) return rc;
    }
    return metadata_->updateLocalSegmentDesc();
}

int HipTransport::unregisterLocalMemoryBatch(
    const std::vector<void*>& addr_list) {
    int first_error = 0;
    for (auto& addr : addr_list) {
        int rc = unregisterLocalMemory(addr, false);
        if (rc && !first_error) first_error = rc;
    }
    int metadata_ret = metadata_->updateLocalSegmentDesc();
    return first_error ? first_error : metadata_ret;
}

void* HipTransport::allocatePinnedLocalMemory(size_t size) {
    int hipDev = 0;
    if (!checkHip(hipGetDevice(&hipDev), "HipTransport: hipGetDevice failed")) {
        return nullptr;
    }

    hipMemAllocationHandleType handle_type = HIPX_MEM_HANDLE_TYPE_POSIX_FD;
#ifdef MC_HIP_HAS_FABRIC_HANDLE
    const bool fabric = fabricExportEnabled() && deviceSupportsFabric(hipDev) &&
                        !(getenv("MC_HIP_ALLOC_FABRIC") &&
                          strcmp(getenv("MC_HIP_ALLOC_FABRIC"), "0") == 0);
#else
    const bool fabric = false;
#endif
    if (fabric) {
#ifdef MC_HIP_HAS_FABRIC_HANDLE
        handle_type = hipMemHandleTypeFabric;
#endif
    } else if (!legacyPosixFdMode()) {
        void* ptr = nullptr;
        if (!checkHip(hipMalloc(&ptr, size),
                      "HipTransport: hipMalloc failed")) {
            return nullptr;
        }
        return ptr;
    }

    size_t granularity = 0;
    hipMemAllocationProp prop = {};
    hipMemGenericAllocationHandle_t handle;
    void* ptr = nullptr;
    int flag = 0;

    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.requestedHandleTypes = handle_type;
    prop.location.id = hipDev;

    hipError_t result = hipDeviceGetAttribute(
        &flag, hipDeviceAttributeVirtualMemoryManagementSupported, hipDev);
    if (!checkHip(result, "HipTransport: hipDeviceGetAttribute failed")) {
        return nullptr;
    }

    if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;

    result = hipMemGetAllocationGranularity(&granularity, &prop,
                                            hipMemAllocationGranularityMinimum);
    if (!checkHip(result,
                  "HipTransport: hipMemGetAllocationGranularity failed")) {
        return nullptr;
    }

    size = (size + granularity - 1) & ~(granularity - 1);
    if (size == 0) size = granularity;

    result = hipMemCreate(&handle, size, &prop, 0);
    if (!checkHip(result, "HipTransport: hipMemCreate failed")) {
        return nullptr;
    }

    result = hipMemAddressReserve((hipDeviceptr_t*)&ptr, size, granularity,
                                  nullptr, 0);
    if (!checkHip(result, "HipTransport: hipMemAddressReserve failed")) {
        (void)hipMemRelease(handle);
        return nullptr;
    }

    result = hipMemMap((hipDeviceptr_t)ptr, size, 0, handle, 0);
    if (!checkHip(result, "HipTransport: hipMemMap failed")) {
        (void)hipMemAddressFree((hipDeviceptr_t)ptr, size);
        (void)hipMemRelease(handle);
        return nullptr;
    }

    int device_count = 0;
    (void)hipGetDeviceCount(&device_count);
    std::vector<hipMemAccessDesc> accessDesc(device_count);
    for (int idx = 0; idx < device_count; ++idx) {
        accessDesc[idx].location.type = hipMemLocationTypeDevice;
        accessDesc[idx].location.id = idx;
        accessDesc[idx].flags = hipMemAccessFlagsProtReadWrite;
    }

    result = hipMemSetAccess((hipDeviceptr_t)ptr, size, accessDesc.data(),
                             device_count);
    if (!checkHip(result, "HipTransport: hipMemSetAccess failed")) {
        (void)hipMemUnmap((hipDeviceptr_t)ptr, size);
        (void)hipMemAddressFree((hipDeviceptr_t)ptr, size);
        (void)hipMemRelease(handle);
        return nullptr;
    }

    return ptr;
}

void HipTransport::freePinnedLocalMemory(void* ptr) {
    if (!ptr) return;
    hipMemGenericAllocationHandle_t handle;
    if (hipMemRetainAllocationHandle(&handle, ptr) != hipSuccess) {
        (void)hipGetLastError();
        (void)hipFree(ptr);
        return;
    }

    hipDeviceptr_t base = nullptr;
    size_t size = 0;
    hipError_t result =
        hipMemGetAddressRange(&base, &size, (hipDeviceptr_t)ptr);
    if (checkHip(result, "HipTransport: hipMemGetAddressRange")) {
        (void)hipMemUnmap(base, size);
        (void)hipMemAddressFree(base, size);
    }

    // One reference from the retain above, one from hipMemCreate.
    (void)hipMemRelease(handle);
    (void)hipMemRelease(handle);
}

}  // namespace mooncake
