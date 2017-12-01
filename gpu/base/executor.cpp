#include "core/base/executor.hpp"


#include "core/base/exception_helpers.hpp"

#include <cuda_runtime.h>
#include "gpu/base/exception.hpp"


namespace gko {


void CpuExecutor::raw_copy_to(const GpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    cudaError_t errcode;
    errcode = cudaMemcpy(dest_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice);
    if (errcode != cudaSuccess) {
        throw CUDA_ERROR(errcode);
    }
}


void GpuExecutor::free(void *ptr) const noexcept { cudaFree(ptr); }


void *GpuExecutor::raw_alloc(size_type num_bytes) const

{
    void *dev_ptr;
    cudaMalloc(&dev_ptr, num_bytes);
    // ENSURE_ALLOCATED(dev_ptr, gpu, num_bytes);
    return dev_ptr;
}


void GpuExecutor::raw_copy_to(const CpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    cudaError_t errcode;
    errcode = cudaMemcpy(dest_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice);
    if (errcode != cudaSuccess) {
        throw CUDA_ERROR(errcode);
    }
}

void GpuExecutor::raw_copy_to(const GpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    cudaError_t errcode;
    errcode = cudaMemcpy(dest_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice);
    if (errcode != cudaSuccess) {
        throw CUDA_ERROR(errcode);
    }
}

}  // namespace gko
