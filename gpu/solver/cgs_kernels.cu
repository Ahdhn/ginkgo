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

#include "core/solver/cgs_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "gpu/base/types.hpp"


namespace gko {
namespace kernels {
namespace gpu {
namespace cgs {


constexpr int default_block_size = 512;

/*
template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void initialize_kernel(
    size_type num_rows, size_type stride, const ValueType *__restrict__ b,
    ValueType *__restrict__ r, ValueType *__restrict__ z,
    ValueType *__restrict__ p, ValueType *__restrict__ q,
    ValueType *__restrict__ alpha, ValueType *__restrict__ beta,
    ValueType *__restrict__ gamma, ValueType *__restrict__ prev_rho,
    ValueType *__restrict__ rho)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tidx < stride) {
        rho[tidx] = zero<ValueType>();
        prev_rho[tidx] = one<ValueType>();
    }

    if (tidx < num_rows * stride) {
        r[tidx] = b[tidx];
        z[tidx] = zero<ValueType>();
        p[tidx] = zero<ValueType>();
        q[tidx] = zero<ValueType>();
    }
}
*/

template <typename ValueType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *r_tld, matrix::Dense<ValueType> *p,
                matrix::Dense<ValueType> *q, matrix::Dense<ValueType> *u,
                matrix::Dense<ValueType> *u_hat,
                matrix::Dense<ValueType> *v_hat, matrix::Dense<ValueType> *t,
                matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *beta,
                matrix::Dense<ValueType> *gamma,
                matrix::Dense<ValueType> *prev_rho,
                matrix::Dense<ValueType> *rho)
{
    NOT_IMPLEMENTED;
    // this is the code from the solver template
    /*
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(b->get_num_rows() * b->get_stride(), block_size.x), 1, 1);

    initialize_kernel<<<grid_size, block_size, 0, 0>>>(
        b->get_num_rows(), b->get_stride(),
    as_cudaValueType(b->get_const_values()), as_cudaValueType(r->get_values()),
    as_cudaValueType(z->get_values()), as_cudaValueType(p->get_values()),
    as_cudaValueType(q->get_values()), as_cudaValueType(prev_rho->get_values()),
    as_cudaValueType(rho->get_values()));
    */
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_INITIALIZE_KERNEL);

/*
template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_2_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type x_stride, ValueType *__restrict__ x, ValueType *__restrict__ r,
    const ValueType *__restrict__ p, const ValueType *__restrict__ q,
    const ValueType *__restrict__ beta, const ValueType *__restrict__ rho)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    const auto row = tidx / stride;
    const auto col = tidx % stride;

    if (col >= num_cols || tidx >= num_rows * num_cols) {
        return;
    }
    if (beta[col] != zero<ValueType>()) {
        const auto tmp = rho[col] / beta[col];
        x[row * x_stride + col] += tmp * p[tidx];
        r[tidx] -= tmp * q[tidx];
    }
}
*/

template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *u,
            matrix::Dense<ValueType> *p, const matrix::Dense<ValueType> *q,
            matrix::Dense<ValueType> *beta, const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *rho_prev)
{
    NOT_IMPLEMENTED;
    // this is the code from the solver template
    /*
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(p->get_num_rows() * p->get_stride(), block_size.x), 1, 1);

    step_2_kernel<<<grid_size, block_size, 0, 0>>>(
        p->get_num_rows(), p->get_num_cols(), p->get_stride(), x->get_stride(),
        as_cudaValueType(x->get_values()), as_cudaValueType(r->get_values()),
        as_cudaValueType(p->get_const_values()),
        as_cudaValueType(q->get_const_values()),
        as_cudaValueType(beta->get_const_values()),
        as_cudaValueType(rho->get_const_values()));
    */
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *u,
            const matrix::Dense<ValueType> *v_hat, matrix::Dense<ValueType> *q,
            matrix::Dense<ValueType> *t, matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *gamma)
{
    NOT_IMPLEMENTED;
    // this is the code from the solver template
    /*
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(p->get_num_rows() * p->get_stride(), block_size.x), 1, 1);

    step_2_kernel<<<grid_size, block_size, 0, 0>>>(
        p->get_num_rows(), p->get_num_cols(), p->get_stride(), x->get_stride(),
        as_cudaValueType(x->get_values()), as_cudaValueType(r->get_values()),
        as_cudaValueType(p->get_const_values()),
        as_cudaValueType(q->get_const_values()),
        as_cudaValueType(beta->get_const_values()),
        as_cudaValueType(rho->get_const_values()));
    */
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *t,
            const matrix::Dense<ValueType> *u_hat, matrix::Dense<ValueType> *r,
            matrix::Dense<ValueType> *x, const matrix::Dense<ValueType> *alpha)
{
    NOT_IMPLEMENTED;
    // this is the code from the solver template
    /*
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(p->get_num_rows() * p->get_stride(), block_size.x), 1, 1);

    step_2_kernel<<<grid_size, block_size, 0, 0>>>(
        p->get_num_rows(), p->get_num_cols(), p->get_stride(), x->get_stride(),
        as_cudaValueType(x->get_values()), as_cudaValueType(r->get_values()),
        as_cudaValueType(p->get_const_values()),
        as_cudaValueType(q->get_const_values()),
        as_cudaValueType(beta->get_const_values()),
        as_cudaValueType(rho->get_const_values()));
    */
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_3_KERNEL);


}  // namespace cgs
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
