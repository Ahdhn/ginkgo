#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"


namespace gko {


void CpuExecutor::raw_copy_to(const GpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
    NOT_COMPILED(gpu);


void GpuExecutor::free(void *ptr) const noexcept
{
    // Free must never fail, as it can be called in destructors.
    // If the nvidia module was not compiled, the library couldn't have
    // allocated the memory, so there is no need to deallocate it.
}


void *GpuExecutor::raw_alloc(size_type num_bytes) const NOT_COMPILED(nvidia);


void GpuExecutor::raw_copy_to(const CpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
    NOT_COMPILED(gpu);


void GpuExecutor::raw_copy_to(const GpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
    NOT_COMPILED(gpu);


}  // namespace gko


#define GKO_HOOK_MODULE gpu
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE
