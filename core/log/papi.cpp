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


#include <ginkgo/core/log/papi.hpp>


#include "core/base/dim.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace log {


template <typename ValueType>
size_type Papi<ValueType>::logger_count = 0;


template <typename ValueType>
void Papi<ValueType>::on_allocation_started(const Executor *exec,
                                            const size_type &num_bytes) const
{
    add_to_map(exec, allocation_started, "allocation_started");
}


template <typename ValueType>
void Papi<ValueType>::on_allocation_completed(const Executor *exec,
                                              const size_type &num_bytes,
                                              const uintptr &location) const
{
    add_to_map(exec, allocation_completed, "allocation_completed");
}


template <typename ValueType>
void Papi<ValueType>::on_free_started(const Executor *exec,
                                      const uintptr &location) const
{
    add_to_map(exec, free_started, "free_started");
}


template <typename ValueType>
void Papi<ValueType>::on_free_completed(const Executor *exec,
                                        const uintptr &location) const
{
    add_to_map(exec, free_completed, "free_completed");
}


template <typename ValueType>
void Papi<ValueType>::on_copy_started(const Executor *from, const Executor *to,
                                      const uintptr &location_from,
                                      const uintptr &location_to,
                                      const size_type &num_bytes) const
{
    add_to_map(from, copy_started_from, "copy_started_from");
    add_to_map(to, copy_started_to, "copy_started_to");
}


template <typename ValueType>
void Papi<ValueType>::on_copy_completed(const Executor *from,
                                        const Executor *to,
                                        const uintptr &location_from,
                                        const uintptr &location_to,
                                        const size_type &num_bytes) const
{
    add_to_map(from, copy_completed_from, "copy_completed_from");
    add_to_map(to, copy_completed_to, "copy_completed_to");
}


template <typename ValueType>
void Papi<ValueType>::on_operation_launched(const Executor *exec,
                                            const Operation *operation) const
{
    add_to_map(exec, operation_launched, "operation_launched");
}


template <typename ValueType>
void Papi<ValueType>::on_operation_completed(const Executor *exec,
                                             const Operation *operation) const
{
    add_to_map(exec, operation_completed, "operation_completed");
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_create_started(
    const Executor *exec, const PolymorphicObject *po) const
{
    add_to_map(exec, po_create_started, "po_create_started");
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_create_completed(
    const Executor *exec, const PolymorphicObject *input,
    const PolymorphicObject *output) const
{
    add_to_map(exec, po_create_completed, "po_create_completed");
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_copy_started(
    const Executor *exec, const PolymorphicObject *from,
    const PolymorphicObject *to) const
{
    add_to_map(exec, po_copy_started, "po_copy_started");
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_copy_completed(
    const Executor *exec, const PolymorphicObject *from,
    const PolymorphicObject *to) const
{
    add_to_map(exec, po_copy_completed, "po_copy_completed");
}


template <typename ValueType>
void Papi<ValueType>::on_polymorphic_object_deleted(
    const Executor *exec, const PolymorphicObject *po) const
{
    add_to_map(exec, po_deleted, "po_deleted");
}


template <typename ValueType>
void Papi<ValueType>::on_linop_apply_started(const LinOp *A, const LinOp *b,
                                             const LinOp *x) const
{
    add_to_map(A, apply_started, "apply_started");
}


template <typename ValueType>
void Papi<ValueType>::on_linop_apply_completed(const LinOp *A, const LinOp *b,
                                               const LinOp *x) const
{
    add_to_map(A, apply_completed, "apply_completed");
}


template <typename ValueType>
void Papi<ValueType>::on_linop_advanced_apply_started(const LinOp *A,
                                                      const LinOp *alpha,
                                                      const LinOp *b,
                                                      const LinOp *beta,
                                                      const LinOp *x) const
{
    add_to_map(A, advanced_apply_started, "advanced_apply_started");
}


template <typename ValueType>
void Papi<ValueType>::on_linop_advanced_apply_completed(const LinOp *A,
                                                        const LinOp *alpha,
                                                        const LinOp *b,
                                                        const LinOp *beta,
                                                        const LinOp *x) const
{
    add_to_map(A, advanced_apply_completed, "advanced_apply_completed");
}


template <typename ValueType>
void Papi<ValueType>::on_linop_factory_generate_started(
    const LinOpFactory *factory, const LinOp *input) const
{
    add_to_map(factory, factory_generate_started, "factory_generate_started");
}


template <typename ValueType>
void Papi<ValueType>::on_linop_factory_generate_completed(
    const LinOpFactory *factory, const LinOp *input, const LinOp *output) const
{
    add_to_map(factory, factory_generate_completed,
               "factory_generate_completed");
}


template <typename ValueType>
void Papi<ValueType>::on_criterion_check_completed(
    const stop::Criterion *criterion, const size_type &num_iterations,
    const LinOp *residual, const LinOp *residual_norm, const LinOp *solution,
    const uint8 &stoppingId, const bool &setFinalized,
    const Array<stopping_status> *status, const bool &oneChanged,
    const bool &converged) const
{
    using Vector = matrix::Dense<ValueType>;
    double residual_norm_d = 0.0;
    if (residual_norm != nullptr) {
        auto dense_res_norm = as<Vector>(residual_norm);
        residual_norm_d =
            static_cast<double>(std::real(dense_res_norm->at(0, 0)));
    } else if (residual != nullptr) {
        auto tmp_res_norm = Vector::create(residual->get_executor(),
                                           dim<2>{1, residual->get_size()[1]});
        auto dense_r = as<Vector>(residual);
        dense_r->compute_norm2(tmp_res_norm.get());
        residual_norm_d = static_cast<double>(std::real(dense_r->at(0, 0)));
    }

    auto &map = criterion_check_completed;
    const auto tmp = reinterpret_cast<uintptr>(criterion);
    if (map.find(tmp) == map.end()) {
        map[tmp] = 0.0;
    }
    auto &value = map[tmp];
    if (!value) {
        std::ostringstream oss;
        oss << "criterion_check_completed"
            << "::" << tmp;
        papi_sde_register_counter(this->papi_handle, oss.str().c_str(),
                                  PAPI_SDE_RO | PAPI_SDE_INSTANT,
                                  PAPI_SDE_long_long, &value);
    }
    map[tmp] = residual_norm_d;
}


template <typename ValueType>
void Papi<ValueType>::on_iteration_complete(const LinOp *solver,
                                            const size_type &num_iterations,
                                            const LinOp *residual,
                                            const LinOp *solution,
                                            const LinOp *residual_norm) const
{
    add_to_map(solver, iteration_complete, "iteration_complete");
}


#define GKO_DECLARE_PAPI(_type) class Papi<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PAPI);
#undef GKO_DECLARE_PAPI


}  // namespace log
}  // namespace gko
