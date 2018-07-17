/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/sellp_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace sellp {


namespace {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(matrix::default_slice_size) void spmv_kernel(
    size_type num_rows, const size_type *__restrict__ slice_lengths,
    const size_type *__restrict__ slice_sets, const ValueType *__restrict__ a,
    const IndexType *__restrict__ col, const ValueType *__restrict__ b,
    ValueType *__restrict__ c)
{
    const auto idx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    ValueType val = 0;
    IndexType ind = 0;
    if (idx < num_rows) {
        for (size_type i = 0; i < slice_lengths[blockIdx.x]; i++) {
            ind = threadIdx.x + (slice_sets[blockIdx.x] + i) * blockDim.x;
            val += a[ind] * b[col[ind]];
        }
        c[idx + blockIdx.y * num_rows] = val;
    }
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Sellp<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    const dim3 blockSize(matrix::default_slice_size);
    const dim3 gridSize(
        ceildiv(a->get_size().num_rows, matrix::default_slice_size),
        b->get_size().num_cols);

    spmv_kernel<<<gridSize, blockSize, 0, 0>>>(
        a->get_size().num_rows, a->get_const_slice_lengths(),
        a->get_const_slice_sets(), as_cuda_type(a->get_const_values()),
        a->get_const_col_idxs(), as_cuda_type(b->get_const_values()),
        as_cuda_type(c->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_SPMV_KERNEL);


namespace {


template <typename ValueType, typename IndexType>
__global__
    __launch_bounds__(matrix::default_slice_size) void advanced_spmv_kernel(
        size_type num_rows, const size_type *__restrict__ slice_lengths,
        const size_type *__restrict__ slice_sets,
        const ValueType *__restrict__ alpha, const ValueType *__restrict__ a,
        const IndexType *__restrict__ col, const ValueType *__restrict__ b,
        const ValueType *__restrict__ beta, ValueType *__restrict__ c)
{
    const auto idx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    ValueType val = 0;
    IndexType ind = 0;
    if (idx < num_rows) {
        for (size_type i = 0; i < slice_lengths[blockIdx.x]; i++) {
            ind = threadIdx.x + (slice_sets[blockIdx.x] + i) * blockDim.x;
            val += alpha[0] * a[ind] * b[col[ind]];
        }
        c[idx + blockIdx.y * num_rows] = beta[0] * c[idx] + val;
    }
}


}  // namespace


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Sellp<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    const dim3 blockSize(matrix::default_slice_size);
    const dim3 gridSize(
        ceildiv(a->get_size().num_rows, matrix::default_slice_size),
        b->get_size().num_cols);

    advanced_spmv_kernel<<<gridSize, blockSize>>>(
        a->get_size().num_rows, a->get_const_slice_lengths(),
        a->get_const_slice_sets(), as_cuda_type(alpha->get_const_values()),
        as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
        as_cuda_type(b->get_const_values()),
        as_cuda_type(beta->get_const_values()), as_cuda_type(c->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const CudaExecutor> exec, matrix::Dense<ValueType> *result,
    const matrix::Sellp<ValueType, IndexType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_CONVERT_TO_DENSE_KERNEL);


}  // namespace sellp
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
