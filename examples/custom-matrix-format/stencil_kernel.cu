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
__device__ __forceinline__  T pitch(const T i, const T j, const T k, const T c, const T dim_x, const T dim_y, const T dim_z)
{
    return c * dim_x * dim_y * dim_z + k * dim_y * dim_z + j * dim_y + i;
}


// a parallel CUDA kernel that computes the application of a 3 point stencil
template <typename ValueType, typename BoundaryType>
__global__ void stencil_kernel_impl(std::size_t size, BoundaryType *bd,
                                    const ValueType *input, ValueType *output,
                                    std::size_t dimx, std::size_t dimy,
                                    std::size_t dimz, bool init)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= size) {
        return;
    }
    std::size_t k = thread_id / (dimy * dimx);
    const std::size_t blah = thread_id - (k * dimx * dimy);
    std::size_t j = blah / dimx;
    std::size_t i = blah % dimx;

    // assert(pitch(i,j,k,dimx, dimy, dimz) == thread_id);

    // printf("\n i= %d, j= %d, k= %d, output= %f, input= %f, bd= %f", i, j, k,
    //       output[thread_id], input[thread_id], bd[thread_id]);

    auto center_pitch = pitch(i, j, k, std::size_t(0), dimx, dimy, dimz);

    if(!init){
        if (k == 0 || k == dimz - 1) {
            bd[center_pitch] = 0;
        } else {
            bd[center_pitch] = 1;
        }
    }

    const ValueType center = input[center_pitch];

    if (bd[center_pitch] == 0) {
        if (!init) {
            output[center_pitch] = 0;
        } else {
            output[center_pitch] = center;
        }

    } else {
        ValueType sum = 0.0;
        int numNeighb = 0;

        if (i > 0) {
            ++numNeighb;
            sum += input[pitch(i - 1, j, k, std::size_t(0), dimx, dimy, dimz)];
        }

        if (j > 0) {
            ++numNeighb;
            sum += input[pitch(i, j - 1, k, std::size_t(0), dimx, dimy, dimz)];
        }

        if (k > 0) {
            ++numNeighb;
            sum += input[pitch(i, j, k - 1, std::size_t(0), dimx, dimy, dimz)];
        }

        if (i < dimx - 1) {
            ++numNeighb;
            sum += input[pitch(i + 1, j, k, std::size_t(0), dimx, dimy, dimz)];
        }

        if (j < dimy - 1) {
            ++numNeighb;
            sum += input[pitch(i, j + 1, k, std::size_t(0), dimx, dimy, dimz)];
        }

        if (k < dimz - 1) {
            ++numNeighb;
            sum += input[pitch(i, j, k + 1, std::size_t(0), dimx, dimy, dimz)];
        }
        const ValueType invh2 = ValueType(1.0);
        output[center_pitch] =
            (-sum + static_cast<ValueType>(numNeighb) * center) * invh2;
    }
}


}  // namespace


template <typename ValueType, typename BoundaryType>
void stencil_kernel(std::size_t size, BoundaryType *bd,
                    const ValueType *input, ValueType *output, std::size_t dimx,
                    std::size_t dimy, std::size_t dimz, bool init)
{
    constexpr auto block_size = 512;
    const auto grid_size = (size + block_size - 1) / block_size;
    stencil_kernel_impl<ValueType, BoundaryType><<<grid_size, block_size>>>(
        size, bd, input, output, dimx, dimy, dimz, init);
}

template void stencil_kernel<float, float>(std::size_t size, float *bd,
                                           const float *input, float *output,
                                           std::size_t dimx, std::size_t dimy,
                                           std::size_t dimz, bool init);
template void stencil_kernel<double, float>(std::size_t size, float *bd,
                                            const double *input, double *output,
                                            std::size_t dimx, std::size_t dimy,
                                            std::size_t dimz, bool init);
