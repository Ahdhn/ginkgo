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

#define GKO_HOOK_MODULE cpu
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE

#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/matrix/csr_kernels.hpp"


namespace gko {


namespace kernels {


namespace cpu {


template <typename ValueType>
GKO_DECLARE_GEMM_KERNEL(ValueType)
NOT_COMPILED(cpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GEMM_KERNEL);

template <typename ValueType>
GKO_DECLARE_SCAL_KERNEL(ValueType)
NOT_COMPILED(cpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_SCAL_KERNEL);

template <typename ValueType>
GKO_DECLARE_AXPY_KERNEL(ValueType)
NOT_COMPILED(cpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_AXPY_KERNEL);

template <typename ValueType>
GKO_DECLARE_DOT_KERNEL(ValueType)
NOT_COMPILED(cpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DOT_KERNEL);


namespace csr {


template <typename ValueType, typename IndexType>
GKO_DECLARE_CSR_SPMV_KERNEL(ValueType, IndexType)
NOT_COMPILED(cpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPMV_KERNEL);

template <typename ValueType, typename IndexType>
GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL(ValueType, IndexType)
NOT_COMPILED(cpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL);


}  // namespace csr
}  // namespace cpu
}  // namespace kernels
}  // namespace gko
