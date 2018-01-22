#include "core/solver/bicgstab_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "gpu/base/types.hpp"


namespace gko {
namespace kernels {
namespace gpu {
namespace bicgstab {


struct size {
    size_type num_rows_;
    size_type num_cols_;
    constexpr size_type get_num_rows() const noexcept { return num_rows_; }
    constexpr size_type get_num_cols() const noexcept { return num_cols_; }
};


inline int64 ceildiv(int64 a, int64 b) { return (a + b - 1) / b; }


template <typename ValueType>
__global__ void initialize_kernel(size_type m, size_type n, size_type lda,
                                  const ValueType *b, ValueType *r,
                                  ValueType *z, ValueType *p, ValueType *v,
                                  ValueType *t, ValueType *y, ValueType *rr,
                                  ValueType *s, ValueType *prev_rho,
                                  ValueType *rho, ValueType *beta,
                                  ValueType *alpha, ValueType *omega)
{
    const size_type tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < n) {
        rho[tidx] = one<ValueType>();
        alpha[tidx] = one<ValueType>();
        beta[tidx] = one<ValueType>();
        omega[tidx] = one<ValueType>();
        prev_rho[tidx] = one<ValueType>();
    }

    if (tidx < m * lda) {
        r[tidx] = b[tidx];
        rr[tidx] = b[tidx];
        z[tidx] = zero<ValueType>();
        p[tidx] = zero<ValueType>();
        v[tidx] = zero<ValueType>();
        t[tidx] = zero<ValueType>();
        s[tidx] = zero<ValueType>();
        y[tidx] = zero<ValueType>();
    }
}


template <typename ValueType>
void initialize(const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *rr, matrix::Dense<ValueType> *y,
                matrix::Dense<ValueType> *s, matrix::Dense<ValueType> *t,
                matrix::Dense<ValueType> *z, matrix::Dense<ValueType> *v,
                matrix::Dense<ValueType> *p, matrix::Dense<ValueType> *prev_rho,
                matrix::Dense<ValueType> *rho, matrix::Dense<ValueType> *alpha,
                matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *omega)
{
    constexpr int block_size_x = 512;
    const dim3 block_size(block_size_x, 1, 1);
    const dim3 grid_size(
        ceildiv(b->get_num_rows() * b->get_padding(), block_size.x), 1, 1);

    initialize_kernel<<<grid_size, block_size, 0, 0>>>(
        b->get_num_rows(), b->get_num_cols(), b->get_padding(),
        as_cuda_type(b->get_const_values()), as_cuda_type(r->get_values()),
        as_cuda_type(z->get_values()), as_cuda_type(p->get_values()),
        as_cuda_type(v->get_values()), as_cuda_type(t->get_values()),
        as_cuda_type(y->get_values()), as_cuda_type(rr->get_values()),
        as_cuda_type(s->get_values()), as_cuda_type(prev_rho->get_values()),
        as_cuda_type(rho->get_values()), as_cuda_type(beta->get_values()),
        as_cuda_type(alpha->get_values()), as_cuda_type(omega->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_INITIALIZE_KERNEL);


template <typename ValueType>
__global__ void step_1_kernel(size_type m, size_type n, size_type lda,
                              ValueType *p, const ValueType *r,
                              const ValueType *v, const ValueType *rho,
                              const ValueType *prev_rho, const ValueType *alpha,
                              const ValueType *omega)
{
    const size_type tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_type col = tidx % lda;
    ValueType tmp = zero<ValueType>();


    if (tidx < m * lda) {
        tmp = rho[col] / prev_rho[col] * alpha[col] / omega[col];
        p[tidx] = (tmp == zero<ValueType>())
                      ? r[tidx]
                      : r[tidx] + tmp * (p[tidx] - omega[col] * v[tidx]);
    }
}


template <typename ValueType>
void step_1(const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *p,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *prev_rho,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *omega)
{
    constexpr int block_size_x = 512;
    const dim3 block_size(block_size_x, 1, 1);
    const dim3 grid_size(
        ceildiv(p->get_num_rows() * p->get_padding(), block_size.x), 1, 1);

    step_1_kernel<<<grid_size, block_size, 0, 0>>>(
        p->get_num_rows(), p->get_num_cols(), p->get_padding(),
        as_cuda_type(p->get_values()), as_cuda_type(r->get_const_values()),
        as_cuda_type(v->get_const_values()),
        as_cuda_type(rho->get_const_values()),
        as_cuda_type(prev_rho->get_const_values()),
        as_cuda_type(alpha->get_const_values()),
        as_cuda_type(omega->get_const_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_1_KERNEL);


template <typename ValueType>
__global__ void step_2_kernel(size_type m, size_type n, size_type lda,
                              ValueType *s, const ValueType *r,
                              const ValueType *v, ValueType *alpha,
                              const ValueType *beta, const ValueType *rho)
{
    const size_type tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_type col = tidx % lda;
    // ValueType tmp = zero<ValueType>();
    if (tidx < n) {
        alpha[n] = rho[n] / beta[n];
    }
    __syncthreads();
    if (tidx < m * lda) {
        alpha[col] = rho[col] / beta[col];
        s[tidx] = (alpha[col] == zero<ValueType>())
                      ? r[tidx]
                      : r[tidx] - alpha[col] * v[tidx];
    }
}


template <typename ValueType>
void step_2(const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *s,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *rho,
            matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *beta)
{
    constexpr int block_size_x = 512;
    const dim3 block_size(block_size_x, 1, 1);
    const dim3 grid_size(
        ceildiv(s->get_num_rows() * s->get_padding(), block_size.x), 1, 1);

    step_2_kernel<<<grid_size, block_size, 0, 0>>>(
        s->get_num_rows(), s->get_num_cols(), s->get_padding(),
        as_cuda_type(s->get_values()), as_cuda_type(r->get_const_values()),
        as_cuda_type(v->get_const_values()), as_cuda_type(alpha->get_values()),
        as_cuda_type(beta->get_const_values()),
        as_cuda_type(rho->get_const_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_2_KERNEL);


template <typename ValueType>
__global__ void step_3_kernel(size_type m, size_type n, size_type lda,
                              ValueType *x, ValueType *r, const ValueType *y,
                              const ValueType *z, const ValueType *s,
                              const ValueType *t, ValueType *omega,
                              const ValueType *alpha, const ValueType *beta)
{
    const size_type tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_type col = tidx % lda;
    // ValueType tmp = zero<ValueType>();
    if (tidx < n) {
        omega[n] = omega[n] / beta[n];
    }
    __syncthreads();
    if (tidx < m * lda) {
        omega[col] = omega[col] / beta[col];
        // x[tidx] = (omega[col] == zero<ValueType>()) ? x[tidx] : x[tidx] +
        // alpha[col] * y[tidx] + omega[col]*z[tidx];
        x[tidx] = x[tidx] + alpha[col] * y[tidx] + omega[col] * z[tidx];
        // r[tidx] = (omega[col] == zero<ValueType>()) ? r[tidx] : r[tidx] - tmp
        // * q[tidx];
        r[tidx] = s[tidx] - omega[col] * t[tidx];
    }
}

template <typename ValueType>
void step_3(matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *r,
            const matrix::Dense<ValueType> *s,
            const matrix::Dense<ValueType> *t,
            const matrix::Dense<ValueType> *y,
            const matrix::Dense<ValueType> *z,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *beta,
            matrix::Dense<ValueType> *omega)
{
    constexpr int block_size_x = 512;
    const dim3 block_size(block_size_x, 1, 1);
    const dim3 grid_size(
        ceildiv(x->get_num_rows() * x->get_padding(), block_size.x), 1, 1);

    step_3_kernel<<<grid_size, block_size, 0, 0>>>(
        x->get_num_rows(), x->get_num_cols(), x->get_padding(),
        as_cuda_type(x->get_values()), as_cuda_type(r->get_values()),
        as_cuda_type(y->get_const_values()),
        as_cuda_type(z->get_const_values()),
        as_cuda_type(s->get_const_values()),
        as_cuda_type(t->get_const_values()), as_cuda_type(omega->get_values()),
        as_cuda_type(alpha->get_const_values()),
        as_cuda_type(beta->get_const_values()));
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_3_KERNEL);


}  // namespace bicgstab
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
