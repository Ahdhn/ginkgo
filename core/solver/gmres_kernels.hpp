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

#ifndef GKO_CORE_SOLVER_GMRES_KERNELS_HPP_
#define GKO_CORE_SOLVER_GMRES_KERNELS_HPP_


#include "core/base/array.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"
#include "core/matrix/dense.hpp"
#include "core/stop/stopping_status.hpp"

namespace gko {
namespace kernels {
namespace gmres {


#define GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL(_type)                          \
    void initialize_1(std::shared_ptr<const DefaultExecutor> exec,            \
                      const matrix::Dense<_type> *b, matrix::Dense<_type> *r, \
                      matrix::Dense<_type> *e1, matrix::Dense<_type> *sn,     \
                      matrix::Dense<_type> *cs, matrix::Dense<_type> *b_norm, \
                      Array<size_type> *iter_nums,                            \
                      Array<stopping_status> *stop_status)

#define GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL(_type, _accessor)    \
    void initialize_2(std::shared_ptr<const DefaultExecutor> exec, \
                      const matrix::Dense<_type> *r,               \
                      matrix::Dense<_type> *r_norm,                \
                      matrix::Dense<_type> *beta, _accessor range_Q)


#define GKO_DECLARE_GMRES_STEP_1_KERNEL(_type, _accessor)                     \
    void step_1(                                                              \
        std::shared_ptr<const DefaultExecutor> exec, matrix::Dense<_type> *q, \
        matrix::Dense<_type> *sn, matrix::Dense<_type> *cs,                   \
        matrix::Dense<_type> *beta, _accessor range_Q, _accessor range_H_k,   \
        matrix::Dense<_type> *r_norm, const matrix::Dense<_type> *b_norm,     \
        const size_type iter_id, const Array<stopping_status> *stop_status)


#define GKO_DECLARE_GMRES_STEP_2_KERNEL(_type, _accessor)                   \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,                \
                const matrix::Dense<_type> *beta, _accessor range_H,        \
                const Array<size_type> *iter_nums, matrix::Dense<_type> *y, \
                _accessor range_Q, matrix::Dense<_type> *x)


#define DECLARE_ALL_AS_TEMPLATES                                    \
    template <typename ValueType>                                   \
    GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL(ValueType);               \
    template <typename ValueType, typename AccessorType>            \
    GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL(ValueType, AccessorType); \
    template <typename ValueType, typename AccessorType>            \
    GKO_DECLARE_GMRES_STEP_1_KERNEL(ValueType, AccessorType);       \
    template <typename ValueType, typename AccessorType>            \
    GKO_DECLARE_GMRES_STEP_2_KERNEL(ValueType, AccessorType)


}  // namespace gmres


namespace omp {
namespace gmres {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace gmres
}  // namespace omp


namespace cuda {
namespace gmres {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace gmres
}  // namespace cuda


namespace reference {
namespace gmres {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace gmres
}  // namespace reference


#undef DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_GMRES_KERNELS_HPP
