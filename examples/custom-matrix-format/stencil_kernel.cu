/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <cstdlib>

#include <ginkgo/ginkgo.hpp>


namespace {

template <typename T>
__device__ __forceinline__ T pitch(const T i, const T j, const T k, const T dim_x, const T dim_y, const T dim_z)
{    
    return k * dim_y * dim_z + j * dim_y + i;
}

// a parallel CUDA kernel that computes the application of a 3 point stencil
template <typename ValueType, typename BoundaryType>
__global__ void stencil_kernel_impl(std::size_t size, const BoundaryType *bd,
                                    const ValueType *b, ValueType *x, std::size_t dimx, std::size_t dimy, std::size_t dimz)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= size) {
        return;
    }
    int k = thread_id / (dimy * dimx); 
    const int blah = thread_id - (k * dimx * dimy);
    int j = blah / dimx;
    int i = blah % dimx;
    
    //assert(pitch(i,j,k,dimx, dimy, dimz) == thread_id);

    //printf("\n i= %d, j= %d, k= %d x= %f, b= %f, bd= %f", i, j, k, 
    //            x[thread_id], b[thread_id], bd[thread_id]);
    printf("\n i= %d, j= %d, k= %d, x= %f, b= %f", i, j, k, x[thread_id], b[thread_id]);

    /*auto result = coefs[1] * b[thread_id];
    if (thread_id > 0) {
        result += coefs[0] * b[thread_id - 1];
    }
    if (thread_id < size - 1) {
        result += coefs[2] * b[thread_id + 1];
    }
    x[thread_id] = result;*/
}


}  // namespace


template <typename ValueType, typename BoundaryType>
void stencil_kernel(std::size_t size, const BoundaryType *bd, const ValueType *b, ValueType *x,
                    std::size_t dimx, std::size_t dimy, std::size_t dimz)
{
    constexpr auto block_size = 512;
    const auto grid_size = (size + block_size - 1) / block_size;
    stencil_kernel_impl<ValueType, BoundaryType><<<grid_size, block_size>>>(size, bd, b, x, dimx, dimy, dimz);
}

template void stencil_kernel<float, float>(std::size_t size, const float *bd, const float *b, float *x, std::size_t, std::size_t, std::size_t);
template void stencil_kernel<double, float>(std::size_t size, const float *bd, const double *b, double *x, std::size_t, std::size_t, std::size_t);
