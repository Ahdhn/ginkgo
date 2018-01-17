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

#include "core/solver/bicgstab.hpp"

#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/utils.hpp"
#include "core/solver/bicgstab_kernels.hpp"

namespace gko {
namespace solver {


namespace {


template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(initialize, bicgstab::initialize<ValueType>);
    GKO_REGISTER_OPERATION(step_1, bicgstab::step_1<ValueType>);
    GKO_REGISTER_OPERATION(step_2, bicgstab::step_2<ValueType>);
    GKO_REGISTER_OPERATION(step_3, bicgstab::step_3<ValueType>);
};

/**
 * Checks whether the required residual goal has been reached or not.
 *
 * @param tau  Residual of the iteration.
 * @param orig_tau  Original residual.
 * @param r  Relative residual goal.
 */
template <typename ValueType>
bool has_converged(const matrix::Dense<ValueType> *tau,
                   const matrix::Dense<ValueType> *orig_tau,
                   remove_complex<ValueType> r)
{
    using std::abs;
    for (int i = 0; i < tau->get_num_rows(); ++i) {
        if (abs(tau->at(i, 0)) >= r * abs(orig_tau->at(i, 0))) {
            return false;
        }
    }
    return true;
}


}  // namespace


template <typename ValueType>
void Bicgstab<ValueType>::copy_from(const LinOp *other)
{
    auto other_bicgstab = as<Bicgstab<ValueType>>(other);
    system_matrix_ = other_bicgstab->get_system_matrix()->clone();
    this->set_dimensions(other);
}


template <typename ValueType>
void Bicgstab<ValueType>::copy_from(std::unique_ptr<LinOp> other)
{
    auto other_bicgstab = as<Bicgstab<ValueType>>(other.get());
    system_matrix_ = std::move(other_bicgstab->get_system_matrix()->clone());
    this->set_dimensions(other.get());
}


template <typename ValueType>
void Bicgstab<ValueType>::apply(const LinOp *b, LinOp *x) const
{
    using std::swap;
    using Vector = matrix::Dense<ValueType>;
    auto dense_b = as<Vector>(b);
    auto dense_x = as<Vector>(x);

    ASSERT_CONFORMANT(system_matrix_, b);
    ASSERT_EQUAL_DIMENSIONS(b, x);

    auto exec = this->get_executor();
    size_type num_vectors = dense_b->get_num_cols();

    auto one_op = Vector::create(exec, {one<ValueType>()});
    auto neg_one_op = Vector::create(exec, {-one<ValueType>()});

    auto r = Vector::create_with_config_of(dense_b);
    auto z = Vector::create_with_config_of(dense_b);
    auto y = Vector::create_with_config_of(dense_b);
    auto v = Vector::create_with_config_of(dense_b);
    auto s = Vector::create_with_config_of(dense_b);
    auto t = Vector::create_with_config_of(dense_b);
    auto p = Vector::create_with_config_of(dense_b);
    auto rr = Vector::create_with_config_of(dense_b);

    auto alpha = Vector::create(exec, 1, dense_b->get_num_cols(), 1);
    auto beta = Vector::create_with_config_of(alpha.get());
    auto prev_rho = Vector::create_with_config_of(alpha.get());
    auto rho = Vector::create_with_config_of(alpha.get());
    auto omega = Vector::create_with_config_of(alpha.get());
    auto tau = Vector::create_with_config_of(alpha.get());

    auto master_tau =
        Vector::create(exec->get_master(), 1, dense_b->get_num_cols(), 1);
    auto starting_tau = Vector::create_with_config_of(master_tau.get());

    // TODO: replace this with automatic merged kernel generator
    exec->run(TemplatedOperation<ValueType>::make_initialize_operation(
        dense_b, r.get(), rr.get(), y.get(), s.get(), t.get(), z.get(), v.get(),
        p.get(), prev_rho.get(), rho.get(), alpha.get(), beta.get(),
        omega.get()));
    // r = dense_b
    // rr = r
    // rho = 1.0
    // omega = 1.0
    // beta = 1.0
    // alpha = 1.0
    // prev_rho = 1.0
    // v = s = t = z = y = p = 0

    // r = b - Ax
    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    // rr = r
    rr->copy_from(r.get());
    // rho = <rr,r>
    // tau = <r,r>
    r->compute_dot(r.get(), tau.get());
    starting_tau->copy_from(tau.get());
    // v = A r
    system_matrix_->apply(r.get(), v.get());
    // prev_rho->copy_from(rho.get());
    for (int iter = 0; iter < max_iters_; ++iter) {
        r->compute_dot(r.get(), tau.get());
        master_tau->copy_from(tau.get());
        if (has_converged(master_tau.get(), starting_tau.get(),
                          rel_residual_goal_)) {
            break;
        }
        // rho = <rr,r>
        rr->compute_dot(r.get(), rho.get());

        exec->run(TemplatedOperation<ValueType>::make_step_1_operation(
            r.get(), p.get(), v.get(), rho.get(), prev_rho.get(), alpha.get(),
            omega.get()));
        // tmp = rho / prev_rho * alpha / omega
        // p = r + tmp * (p - omega * v)
        // TODO//precond_->apply(p.get(), y.get());
        y->copy_from(p.get());
        // v = A y
        system_matrix_->apply(y.get(), v.get());
        rr->compute_dot(v.get(), beta.get());
        exec->run(TemplatedOperation<ValueType>::make_step_2_operation(
            r.get(), s.get(), v.get(), rho.get(), alpha.get(), beta.get()));
        // alpha = rho / beta
        // s = r - alpha * v
        // TODO//precond_->apply(s.get(), z.get());
        z->copy_from(s.get());
        // t = A z
        system_matrix_->apply(z.get(), t.get());
        s->compute_dot(t.get(), omega.get());
        t->compute_dot(t.get(), beta.get());
        exec->run(TemplatedOperation<ValueType>::make_step_3_operation(
            dense_x, r.get(), s.get(), t.get(), y.get(), z.get(), alpha.get(),
            beta.get(), omega.get()));
        // omega = omega / beta
        // x = x + alpha * y + omega * z
        // r = s - omega * t
        swap(prev_rho, rho);
    }
}


template <typename ValueType>
void Bicgstab<ValueType>::apply(const LinOp *alpha, const LinOp *b,
                                const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


template <typename ValueType>
std::unique_ptr<LinOp> Bicgstab<ValueType>::clone_type() const
{
    return std::unique_ptr<Bicgstab>(
        new Bicgstab(this->get_executor(), max_iters_, rel_residual_goal_,
                     system_matrix_->clone_type()));
}


template <typename ValueType>
void Bicgstab<ValueType>::clear()
{
    this->set_dimensions(0, 0, 0);
    max_iters_ = 0;
    rel_residual_goal_ = zero<decltype(rel_residual_goal_)>();
    system_matrix_ = system_matrix_->clone_type();
}


template <typename ValueType>
std::unique_ptr<LinOp> BicgstabFactory<ValueType>::generate(
    std::shared_ptr<const LinOp> base) const
{
    auto bicgstab =
        std::unique_ptr<Bicgstab<ValueType>>(Bicgstab<ValueType>::create(
            this->get_executor(), max_iters_, rel_residual_goal_, base));
    ASSERT_EQUAL_DIMENSIONS(bicgstab->system_matrix_,
                            size(bicgstab->system_matrix_->get_num_cols(),
                                 bicgstab->system_matrix_->get_num_rows()));

    return std::move(bicgstab);
}


#define GKO_DECLARE_BICGSTAB(_type) class Bicgstab<_type>
#define GKO_DECLARE_BICGSTAB_FACTORY(_type) class BicgstabFactory<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_FACTORY);
#undef GKO_DECLARE_BICGSTAB
#undef GKO_DECLARE_BICGSTAB_FACTORY


}  // namespace solver
}  // namespace gko
