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

#ifndef GKO_CORE_SOLVER_IDR_KERNELS_HPP_
#define GKO_CORE_SOLVER_IDR_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


namespace gko {
namespace kernels {
namespace idr {


#define GKO_DECLARE_IDR_INITIALIZE_KERNEL(_type)                   \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,   \
                    const size_type nrhs, matrix::Dense<_type> *m, \
                    matrix::Dense<_type> *subspace_vectors,        \
                    bool deterministic, Array<stopping_status> *stop_status)


#define GKO_DECLARE_IDR_STEP_1_KERNEL(_type)                                 \
    void step_1(                                                             \
        std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,   \
        const size_type k, const matrix::Dense<_type> *m,                    \
        const matrix::Dense<_type> *f, const matrix::Dense<_type> *residual, \
        const matrix::Dense<_type> *g, matrix::Dense<_type> *c,              \
        matrix::Dense<_type> *v, const Array<stopping_status> *stop_status)


#define GKO_DECLARE_IDR_STEP_2_KERNEL(_type)                            \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,            \
                const size_type nrhs, const size_type k,                \
                const matrix::Dense<_type> *omega,                      \
                const matrix::Dense<_type> *preconditioned_vector,      \
                const matrix::Dense<_type> *c, matrix::Dense<_type> *u, \
                const Array<stopping_status> *stop_status)


#define GKO_DECLARE_IDR_STEP_3_KERNEL(_type)                                 \
    void step_3(std::shared_ptr<const DefaultExecutor> exec,                 \
                const size_type nrhs, const size_type k,                     \
                const matrix::Dense<_type> *p, matrix::Dense<_type> *g,      \
                matrix::Dense<_type> *g_k, matrix::Dense<_type> *u,          \
                matrix::Dense<_type> *m, matrix::Dense<_type> *f,            \
                matrix::Dense<_type> *alpha, matrix::Dense<_type> *residual, \
                matrix::Dense<_type> *x,                                     \
                const Array<stopping_status> *stop_status)


#define GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL(_type)                         \
    void compute_omega(                                                     \
        std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,  \
        const remove_complex<_type> kappa, const matrix::Dense<_type> *tht, \
        const matrix::Dense<remove_complex<_type>> *residual_norm,          \
        matrix::Dense<_type> *omega,                                        \
        const Array<stopping_status> *stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES(_export_macro)             \
    template <typename ValueType>                               \
    _export_macro GKO_DECLARE_IDR_INITIALIZE_KERNEL(ValueType); \
    template <typename ValueType>                               \
    _export_macro GKO_DECLARE_IDR_STEP_1_KERNEL(ValueType);     \
    template <typename ValueType>                               \
    _export_macro GKO_DECLARE_IDR_STEP_2_KERNEL(ValueType);     \
    template <typename ValueType>                               \
    _export_macro GKO_DECLARE_IDR_STEP_3_KERNEL(ValueType);     \
    template <typename ValueType>                               \
    _export_macro GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL(ValueType)


}  // namespace idr


namespace omp {
namespace idr {

GKO_DECLARE_ALL_AS_TEMPLATES(GKO_OMP_EXPORT);

}  // namespace idr
}  // namespace omp


namespace cuda {
namespace idr {

GKO_DECLARE_ALL_AS_TEMPLATES(GKO_CUDA_EXPORT);

}  // namespace idr
}  // namespace cuda


namespace reference {
namespace idr {

GKO_DECLARE_ALL_AS_TEMPLATES(GKO_REFERENCE_EXPORT);

}  // namespace idr
}  // namespace reference


namespace hip {
namespace idr {

GKO_DECLARE_ALL_AS_TEMPLATES(GKO_HIP_EXPORT);

}  // namespace idr
}  // namespace hip


namespace dpcpp {
namespace idr {

GKO_DECLARE_ALL_AS_TEMPLATES(GKO_DPCPP_EXPORT);

}  // namespace idr
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_IDR_KERNELS_HPP_
