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

#ifndef GKO_CORE_STOP_RELATIVE_RESIDUAL_NORM_KERNELS_HPP_
#define GKO_CORE_STOP_RELATIVE_RESIDUAL_NORM_KERNELS_HPP_


#include "core/base/array.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace kernels {
namespace relative_residual_norm {

#define GKO_DECLARE_RELATIVE_RESIDUAL_NORM_KERNEL(_type)                     \
    void relative_residual_norm(std::shared_ptr<const DefaultExecutor> exec, \
                                const matrix::Dense<_type> *tau,             \
                                const matrix::Dense<_type> *orig_tau,        \
                                remove_complex<_type> rel_residual_goal,     \
                                Array<bool> *converged, bool *all_converged)


#define DECLARE_ALL_AS_TEMPLATES  \
    template <typename ValueType> \
    GKO_DECLARE_RELATIVE_RESIDUAL_NORM_KERNEL(ValueType)


}  // namespace relative_residual_norm


namespace cpu {
namespace relative_residual_norm {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace relative_residual_norm
}  // namespace cpu


namespace gpu {
namespace relative_residual_norm {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace relative_residual_norm
}  // namespace gpu


namespace reference {
namespace relative_residual_norm {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace relative_residual_norm
}  // namespace reference


#undef DECLARE_ALL_AS_TEMPLATES

}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_STOP_RELATIVE_RESIDUAL_NORM_KERNELS_HPP_
