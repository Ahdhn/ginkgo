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

#include "core/solver/gmres.hpp"


#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/name_demangling.hpp"
#include "core/base/range_accessors.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/dense.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/solver/gmres_kernels.hpp"


namespace gko {
namespace solver {


namespace {


template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(initialize_1, gmres::initialize_1<ValueType>);
};


template <typename... TplArgs>
struct TemplatedOperationRange {
    GKO_REGISTER_OPERATION(initialize_2, gmres::initialize_2<TplArgs...>);
    GKO_REGISTER_OPERATION(step_1, gmres::step_1<TplArgs...>);
    GKO_REGISTER_OPERATION(step_2, gmres::step_2<TplArgs...>);
};


template <typename... TplArgs>
struct TemplatedOperationDenseRange {
    GKO_REGISTER_OPERATION(simple_apply, dense::simple_apply<TplArgs...>);
    GKO_REGISTER_OPERATION(apply, dense::apply<TplArgs...>);
};


}  // namespace


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    ASSERT_IS_SQUARE_MATRIX(system_matrix_);

    this->template log<log::Logger::apply>(GKO_FUNCTION_NAME);

    using std::swap;
    using Vector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    auto r = Vector::create_with_config_of(dense_b);
    auto z = Vector::create_with_config_of(dense_b);
    auto e1 = Vector::create_with_config_of(dense_x);
    auto beta = Vector::create_with_config_of(dense_b);
    auto Q = Vector::create(exec, dim{system_matrix_->get_size().num_cols,
                                      (default_max_num_iterations + 1) *
                                          dense_b->get_size().num_cols});
    auto H = Vector::create(
        exec, dim{default_max_num_iterations + 1,
                  default_max_num_iterations * dense_b->get_size().num_cols});
    auto sn = Vector::create(
        exec, dim{default_max_num_iterations, dense_b->get_size().num_cols});
    auto cs = Vector::create(
        exec, dim{default_max_num_iterations, dense_b->get_size().num_cols});
    auto r_norm = Vector::create(exec, dim{1, dense_b->get_size().num_cols});
    auto b_norm = Vector::create(exec, dim{1, dense_b->get_size().num_cols});

    // Define range accessors for Q and H.
    using row_major_range = gko::range<gko::accessor::row_major<ValueType, 2>>;
    row_major_range range_Q{Q->get_values(), Q->get_size().num_rows,
                            Q->get_size().num_cols, Q->get_stride()};
    row_major_range range_H{H->get_values(), H->get_size().num_rows,
                            H->get_size().num_cols, H->get_stride()};

    bool one_changed{};
    Array<stopping_status> stop_status(this->get_executor(),
                                       dense_b->get_size().num_cols);

    // TODO: replace this with automatic merged kernel generator
    exec->run(TemplatedOperation<ValueType>::make_initialize_1_operation(
        dense_b, r.get(), e1.get(), sn.get(), cs.get(), b_norm.get(),
        &stop_status));
    // r = dense_b
    // sn = cs = 0
    // e1 = {1, 0, ..., 0}

    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    // r = b - Ax

    exec->run(
        TemplatedOperationRange<ValueType,
                                range<accessor::row_major<ValueType, 2>>>::
            make_initialize_2_operation(r.get(), r_norm.get(), beta.get(),
                                        range_Q));
    // beta = {r_norm, 0, ..., 0}
    // Q(:, 1) = r / r_norm

    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, r.get());

    size_type iter = 0;
    for (; iter < default_max_num_iterations; ++iter) {
        preconditioner_->apply(r.get(), z.get());

        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            this->template log<log::Logger::converged>(iter + 1, r.get());
            break;
        }

        // Start Arnoldi function
        auto range_Q_k =
            range_Q(span{0, system_matrix_->get_size().num_rows},
                    span{dense_b->get_size().num_cols * iter,
                         dense_b->get_size().num_cols * (iter + 1)});
        auto range_H_k = range_H(
            span{0, iter + 1}, span{dense_b->get_size().num_cols * iter,
                                    dense_b->get_size().num_cols * (iter + 1)});
        auto q = Vector::create_with_config_of(dense_b);
        exec->run(TemplatedOperationDenseRange<
                  ValueType, range<accessor::row_major<ValueType, 2>>>::
                      make_simple_apply_operation(
                          as<matrix::Dense<ValueType>>(system_matrix_.get()),
                          range_Q_k, q.get()));
        exec->run(
            TemplatedOperationRange<ValueType,
                                    range<accessor::row_major<ValueType, 2>>>::
                make_step_1_operation(q.get(), sn.get(), cs.get(), beta.get(),
                                      range_Q, range_H_k, iter));
        this->template log<log::Logger::iteration_complete>(iter + 1);
    }

    // Solve x
    auto y = Vector::create(
        exec, dim{default_max_num_iterations, dense_b->get_size().num_cols});
    auto range_Q_iter =
        range_Q(span{0, system_matrix_->get_size().num_rows},
                span{0, dense_b->get_size().num_cols * (iter + 1)});
    exec->run(
        TemplatedOperationRange<ValueType,
                                range<accessor::row_major<ValueType, 2>>>::
            make_step_2_operation(beta.get(), range_H, iter, y.get(),
                                  range_Q_iter, dense_x));
}


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                  const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_GMRES(_type) class Gmres<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES);
#undef GKO_DECLARE_GMRES


}  // namespace solver
}  // namespace gko
