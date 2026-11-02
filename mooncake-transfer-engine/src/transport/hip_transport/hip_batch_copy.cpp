// Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "transport/hip_transport/hip_batch_copy.h"

#include <dlfcn.h>
#include <glog/logging.h>
#include <hip/hiprtc.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>

namespace mooncake {

namespace {

struct CopyDesc {
    uint64_t src;
    uint64_t dst;
    uint64_t length;
    uint64_t piece_begin;
};

// Each block copies whole pieces; thread 0 locates the owning descriptor.
const char *kKernelSource = R"(
struct CopyDesc {
    unsigned long long src, dst, length, piece_begin;
};

extern "C" __global__ void mc_batch_copy(const CopyDesc* __restrict__ descs,
                                         unsigned int ndesc,
                                         unsigned long long npieces,
                                         unsigned long long piece_bytes) {
    __shared__ CopyDesc cur;
    for (unsigned long long p = blockIdx.x; p < npieces; p += gridDim.x) {
        if (threadIdx.x == 0) {
            unsigned int lo = 0, hi = ndesc - 1;
            while (lo < hi) {
                unsigned int mid = (lo + hi + 1) >> 1;
                if (descs[mid].piece_begin <= p) lo = mid; else hi = mid - 1;
            }
            cur = descs[lo];
        }
        __syncthreads();
        const unsigned long long off = (p - cur.piece_begin) * piece_bytes;
        const unsigned long long rest = cur.length - off;
        const unsigned long long n = rest < piece_bytes ? rest : piece_bytes;
        const char* s = (const char*)cur.src + off;
        char* d = (char*)cur.dst + off;
        const unsigned int bd = blockDim.x;
        unsigned long long done = 0;
        if ((((unsigned long long)s | (unsigned long long)d) & 15) == 0) {
            const uint4* sv = (const uint4*)s;
            uint4* dv = (uint4*)d;
            const unsigned long long nv = n >> 4;
            unsigned long long i = threadIdx.x;
            for (; i + 3 * bd < nv; i += 4 * bd) {
                uint4 a = sv[i], b = sv[i + bd], c = sv[i + 2 * bd],
                      e = sv[i + 3 * bd];
                dv[i] = a; dv[i + bd] = b; dv[i + 2 * bd] = c;
                dv[i + 3 * bd] = e;
            }
            for (; i < nv; i += bd) dv[i] = sv[i];
            done = nv << 4;
        }
        for (unsigned long long i = done + threadIdx.x; i < n; i += bd)
            d[i] = s[i];
        __syncthreads();
    }
}
)";

struct HiprtcApi {
    decltype(&hiprtcCreateProgram) createProgram = nullptr;
    decltype(&hiprtcCompileProgram) compileProgram = nullptr;
    decltype(&hiprtcGetProgramLogSize) getProgramLogSize = nullptr;
    decltype(&hiprtcGetProgramLog) getProgramLog = nullptr;
    decltype(&hiprtcGetCodeSize) getCodeSize = nullptr;
    decltype(&hiprtcGetCode) getCode = nullptr;
    decltype(&hiprtcDestroyProgram) destroyProgram = nullptr;
    bool ok = false;

    static const HiprtcApi &get() {
        static HiprtcApi api = load();
        return api;
    }

   private:
    static HiprtcApi load() {
        HiprtcApi api;
        std::vector<std::string> candidates;
        // Prefer the hiprtc that ships next to the loaded HIP runtime.
        Dl_info info;
        if (dladdr(reinterpret_cast<void *>(&hipRuntimeGetVersion), &info) &&
            info.dli_fname) {
            std::string path = info.dli_fname;
            auto slash = path.rfind('/');
            if (slash != std::string::npos) {
                candidates.push_back(path.substr(0, slash + 1) +
                                     "libhiprtc.so.7");
                candidates.push_back(path.substr(0, slash + 1) +
                                     "libhiprtc.so");
            }
        }
        candidates.push_back("libhiprtc.so.7");
        candidates.push_back("libhiprtc.so");
        void *lib = nullptr;
        for (const auto &name : candidates) {
            lib = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (lib) break;
        }
        if (!lib) {
            LOG(WARNING) << "HipBatchCopier: libhiprtc not found, copy kernel "
                            "disabled";
            return api;
        }
#define MC_LOAD_HIPRTC(field, sym)                                       \
    api.field = reinterpret_cast<decltype(api.field)>(dlsym(lib, #sym)); \
    if (!api.field) return api;
        MC_LOAD_HIPRTC(createProgram, hiprtcCreateProgram)
        MC_LOAD_HIPRTC(compileProgram, hiprtcCompileProgram)
        MC_LOAD_HIPRTC(getProgramLogSize, hiprtcGetProgramLogSize)
        MC_LOAD_HIPRTC(getProgramLog, hiprtcGetProgramLog)
        MC_LOAD_HIPRTC(getCodeSize, hiprtcGetCodeSize)
        MC_LOAD_HIPRTC(getCode, hiprtcGetCode)
        MC_LOAD_HIPRTC(destroyProgram, hiprtcDestroyProgram)
#undef MC_LOAD_HIPRTC
        api.ok = true;
        return api;
    }
};

bool compileKernel(const std::string &arch, std::vector<char> &code) {
    const auto &rtc = HiprtcApi::get();
    if (!rtc.ok) return false;
    hiprtcProgram prog;
    if (rtc.createProgram(&prog, kKernelSource, "mc_batch_copy.hip", 0, nullptr,
                          nullptr) != HIPRTC_SUCCESS)
        return false;
    std::string arch_opt = "--offload-arch=" + arch;
    const char *opts[] = {arch_opt.c_str(), "-O3"};
    auto rc = rtc.compileProgram(prog, 2, opts);
    if (rc != HIPRTC_SUCCESS) {
        size_t log_size = 0;
        rtc.getProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        if (log_size) rtc.getProgramLog(prog, log.data());
        LOG(ERROR) << "HipBatchCopier: hiprtc compile for " << arch
                   << " failed: " << log;
        rtc.destroyProgram(&prog);
        return false;
    }
    size_t code_size = 0;
    rtc.getCodeSize(prog, &code_size);
    code.resize(code_size);
    rtc.getCode(prog, code.data());
    rtc.destroyProgram(&prog);
    return code_size > 0;
}

int envInt(const char *name, int fallback) {
    const char *env = getenv(name);
    if (!env || !*env) return fallback;
    char *end = nullptr;
    long value = strtol(env, &end, 0);
    if (!end || *end != '\0' || value <= 0) {
        LOG(WARNING) << "Ignoring invalid " << name;
        return fallback;
    }
    return static_cast<int>(value);
}

}  // namespace

struct HipBatchCopier::Context {
    std::mutex mutex;
    int device = -1;
    hipStream_t stream = nullptr;
    CopyDesc *host_table = nullptr;
    CopyDesc *dev_table = nullptr;
    size_t capacity = 0;

    bool reserve(size_t count) {
        if (count <= capacity) return true;
        size_t cap =
            std::max<size_t>(count, std::max<size_t>(256, capacity * 2));
        CopyDesc *host = nullptr, *dev = nullptr;
        if (hipHostMalloc(reinterpret_cast<void **>(&host),
                          cap * sizeof(CopyDesc),
                          hipHostMallocDefault) != hipSuccess)
            return false;
        if (hipMalloc(reinterpret_cast<void **>(&dev),
                      cap * sizeof(CopyDesc)) != hipSuccess) {
            (void)hipHostFree(host);
            return false;
        }
        if (host_table) (void)hipHostFree(host_table);
        if (dev_table) (void)hipFree(dev_table);
        host_table = host;
        dev_table = dev;
        capacity = cap;
        return true;
    }
};

struct HipBatchCopier::DeviceState {
    std::mutex init_mutex;
    bool initialized = false;
    bool ok = false;
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    int num_cus = 0;
    std::vector<std::unique_ptr<Context>> contexts;
    std::atomic<unsigned> next{0};
};

HipBatchCopier::Launch::~Launch() = default;

HipBatchCopier &HipBatchCopier::instance() {
    static HipBatchCopier copier;
    return copier;
}

HipBatchCopier::HipBatchCopier()
    : piece_bytes_(
          static_cast<uint64_t>(envInt("MC_HIP_COPY_PIECE_BYTES", 64 * 1024))),
      threads_per_block_(envInt("MC_HIP_COPY_THREADS", 512)),
      blocks_per_cu_(envInt("MC_HIP_COPY_BLOCKS_PER_CU", 4)),
      max_cus_(envInt("MC_UALOE_COPY_CUS", 0x7fffffff)),
      contexts_per_device_(envInt("MC_HIP_COPY_CONTEXTS", 4)) {
    piece_bytes_ = std::max<uint64_t>(4096, piece_bytes_ & ~uint64_t(15));
    threads_per_block_ = std::min(1024, std::max(64, threads_per_block_));
}

// Process-lifetime singleton: device resources are reclaimed by the runtime
// at exit, and HIP calls during static destruction are unsafe.
HipBatchCopier::~HipBatchCopier() = default;

HipBatchCopier::DeviceState *HipBatchCopier::deviceState(int device) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (device < 0) return nullptr;
    if (devices_.size() <= static_cast<size_t>(device))
        devices_.resize(device + 1);
    if (!devices_[device]) devices_[device] = std::make_unique<DeviceState>();
    return devices_[device].get();
}

bool HipBatchCopier::ready(int device) {
    DeviceState *state = deviceState(device);
    if (!state) return false;
    std::lock_guard<std::mutex> lock(state->init_mutex);
    if (state->initialized) return state->ok;
    state->initialized = true;

    static std::mutex code_mutex;
    static std::map<std::string, std::vector<char>> code_by_arch;

    int prev = -1;
    (void)hipGetDevice(&prev);
    if (hipSetDevice(device) != hipSuccess) return false;
    hipDeviceProp_t prop;
    bool ok = hipGetDeviceProperties(&prop, device) == hipSuccess;
    std::vector<char> *code = nullptr;
    if (ok) {
        std::lock_guard<std::mutex> code_lock(code_mutex);
        auto &entry = code_by_arch[prop.gcnArchName];
        if (entry.empty()) ok = compileKernel(prop.gcnArchName, entry);
        code = &entry;
    }
    if (ok) ok = hipModuleLoadData(&state->module, code->data()) == hipSuccess;
    if (ok)
        ok = hipModuleGetFunction(&state->function, state->module,
                                  "mc_batch_copy") == hipSuccess;
    if (ok) {
        state->num_cus = prop.multiProcessorCount;
        for (int i = 0; i < contexts_per_device_ && ok; ++i) {
            auto ctx = std::make_unique<Context>();
            ctx->device = device;
            ok = hipStreamCreateWithFlags(&ctx->stream, hipStreamNonBlocking) ==
                 hipSuccess;
            if (ok) state->contexts.push_back(std::move(ctx));
        }
    }
    if (prev >= 0) (void)hipSetDevice(prev);
    if (!ok) {
        (void)hipGetLastError();
        LOG(WARNING) << "HipBatchCopier: copy kernel unavailable on device "
                     << device << ", falling back to hipMemcpyAsync";
        return false;
    }
    LOG(INFO) << "HipBatchCopier: device " << device << " (" << prop.gcnArchName
              << ", " << state->num_cus << " CUs) ready, piece=" << piece_bytes_
              << "B, threads=" << threads_per_block_;
    state->ok = true;
    return true;
}

HipBatchCopier::Context *HipBatchCopier::acquire(
    DeviceState &state, std::unique_lock<std::mutex> &lock) {
    const size_t n = state.contexts.size();
    unsigned start = state.next.fetch_add(1, std::memory_order_relaxed);
    for (size_t i = 0; i < n; ++i) {
        Context *ctx = state.contexts[(start + i) % n].get();
        std::unique_lock<std::mutex> attempt(ctx->mutex, std::try_to_lock);
        if (attempt.owns_lock()) {
            lock = std::move(attempt);
            return ctx;
        }
    }
    Context *ctx = state.contexts[start % n].get();
    lock = std::unique_lock<std::mutex>(ctx->mutex);
    return ctx;
}

hipError_t HipBatchCopier::submit(int device, const std::vector<Item> &items,
                                  std::unique_ptr<Launch> &launch) {
    if (!ready(device)) return hipErrorNotSupported;
    DeviceState &state = *deviceState(device);
    launch.reset(new Launch());
    launch->ctx_ = acquire(state, launch->lock_);
    Context &ctx = *launch->ctx_;

    auto fail = [&](hipError_t err) {
        // The staging buffers may still be read by an enqueued upload.
        (void)hipStreamSynchronize(ctx.stream);
        launch.reset();
        return err;
    };
    if (!ctx.reserve(items.size())) return fail(hipErrorOutOfMemory);
    uint64_t pieces = 0;
    uint32_t ndesc = 0;
    for (const auto &item : items) {
        if (item.length == 0) continue;
        CopyDesc &desc = ctx.host_table[ndesc++];
        desc.src = reinterpret_cast<uint64_t>(item.src);
        desc.dst = reinterpret_cast<uint64_t>(item.dst);
        desc.length = item.length;
        desc.piece_begin = pieces;
        pieces += (item.length + piece_bytes_ - 1) / piece_bytes_;
    }
    if (ndesc == 0) return hipSuccess;

    hipError_t err =
        hipMemcpyAsync(ctx.dev_table, ctx.host_table, ndesc * sizeof(CopyDesc),
                       hipMemcpyHostToDevice, ctx.stream);
    if (err != hipSuccess) return fail(err);

    const uint64_t cus =
        static_cast<uint64_t>(std::min(state.num_cus, max_cus_));
    const uint64_t blocks =
        std::max<uint64_t>(1, std::min<uint64_t>(pieces, cus * blocks_per_cu_));
    const CopyDesc *table = ctx.dev_table;
    unsigned long long npieces = pieces;
    unsigned long long piece_bytes = piece_bytes_;
    void *args[] = {&table, &ndesc, &npieces, &piece_bytes};
    err = hipModuleLaunchKernel(state.function, static_cast<unsigned>(blocks),
                                1, 1, threads_per_block_, 1, 1, 0, ctx.stream,
                                args, nullptr);
    return err == hipSuccess ? err : fail(err);
}

hipError_t HipBatchCopier::wait(std::unique_ptr<Launch> &launch) {
    if (!launch || !launch->ctx_) return hipErrorInvalidValue;
    hipError_t err = hipStreamSynchronize(launch->ctx_->stream);
    launch.reset();
    return err;
}

}  // namespace mooncake
