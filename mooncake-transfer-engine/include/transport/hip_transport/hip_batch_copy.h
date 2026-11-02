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

#ifndef HIP_BATCH_COPY_H_
#define HIP_BATCH_COPY_H_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mooncake {

// Gather/scatter copy kernel for device-to-device transfers over mapped
// peer/fabric memory. One H2D descriptor upload and one launch per batch.
// hipMemcpyAsync on imported fabric pointers is far slower than a copy
// kernel and is reported to livelock for large transfers.
class HipBatchCopier {
   public:
    struct Item {
        void *dst;
        const void *src;
        uint64_t length;
    };

    struct Context;

    // Holds a per-device staging context until wait() returns.
    class Launch {
       public:
        ~Launch();
        Launch(const Launch &) = delete;
        Launch &operator=(const Launch &) = delete;

       private:
        friend class HipBatchCopier;
        Launch() = default;
        Context *ctx_ = nullptr;
        std::unique_lock<std::mutex> lock_;
    };

    static HipBatchCopier &instance();

    // Compiles and loads the kernel for `device` on first use.
    bool ready(int device);

    // Enqueues the copies on a stream of `device`; the caller's current
    // device must be `device`.
    hipError_t submit(int device, const std::vector<Item> &items,
                      std::unique_ptr<Launch> &launch);

    hipError_t wait(std::unique_ptr<Launch> &launch);

    ~HipBatchCopier();

   private:
    HipBatchCopier();

    struct DeviceState;
    DeviceState *deviceState(int device);
    Context *acquire(DeviceState &state, std::unique_lock<std::mutex> &lock);

    std::mutex mutex_;
    std::vector<std::unique_ptr<DeviceState>> devices_;
    uint64_t piece_bytes_;
    int threads_per_block_;
    int blocks_per_cu_;
    int max_cus_;
    int contexts_per_device_;
};

}  // namespace mooncake

#endif  // HIP_BATCH_COPY_H_
