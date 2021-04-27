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

#ifndef GKO_PUBLIC_CORE_SOLVER_SOLVER_BASE_HPP_
#define GKO_PUBLIC_CORE_SOLVER_SOLVER_BASE_HPP_


#include <memory>
#include <utility>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {


/**
 * A LinOp implementing this interface can be preconditioned.
 *
 * @ingroup precond
 * @ingroup LinOp
 */
class Preconditionable {
public:
    virtual ~Preconditionable() = default;

    /**
     * Returns the preconditioner operator used by the Preconditionable.
     *
     * @return the preconditioner operator used by the Preconditionable
     */
    virtual std::shared_ptr<const LinOp> get_preconditioner() const;

    /**
     * Sets the preconditioner operator used by the Preconditionable.
     *
     * @param new_precond  the new preconditioner operator used by the
     *                     Preconditionable
     */
    virtual void set_preconditioner(std::shared_ptr<const LinOp> new_precond);

    Preconditionable &operator=(const Preconditionable &other);

    Preconditionable &operator=(Preconditionable &&other);

    Preconditionable();

    Preconditionable(std::shared_ptr<const LinOp> preconditioner);

    Preconditionable(const Preconditionable &);

    Preconditionable(Preconditionable &&);

private:
    std::shared_ptr<const LinOp> preconditioner_;
};


namespace solver {


/**
 * A LinOp implementing this interface stores a system matrix.
 *
 * @ingroup solver
 * @ingroup LinOp
 */
template <typename MatrixType = LinOp>
class SolverBase {
public:
    virtual ~SolverBase() = default;

    /**
     * Returns the system matrix used by the solver.
     *
     * @return the system matrix operator used by the solver
     */
    std::shared_ptr<const MatrixType> get_system_matrix() const
    {
        return system_matrix_;
    }

    SolverBase &operator=(const SolverBase &other)
    {
        auto this_matrix = this->get_system_matrix();
        auto other_matrix = other.get_system_matrix();
        if (other_matrix == nullptr || this_matrix == nullptr ||
            other_matrix->get_executor() == this_matrix->get_executor()) {
            this->system_matrix_ = other_matrix;
        } else {  // both system matrices are non-null with different executors
            this->system_matrix_ =
                clone(this_matrix->get_executor(), other_matrix);
        }
        return *this;
    }

    SolverBase &operator=(SolverBase &&other)
    {
        *this = other;
        other.system_matrix_ = nullptr;
        return *this;
    }

    SolverBase() : SolverBase{nullptr} {}

    SolverBase(std::shared_ptr<const MatrixType> system_matrix)
        : system_matrix_{std::move(system_matrix)}
    {}

    SolverBase(const SolverBase &other) : SolverBase{} { *this = other; }

    SolverBase(SolverBase &&other) : SolverBase{} { *this = std::move(other); }

private:
    std::shared_ptr<const MatrixType> system_matrix_;
};


/**
 * A LinOp implementing this interface stores a system matrix and stopping
 * criterion factory.
 *
 * @ingroup solver
 * @ingroup LinOp
 */
class IterativeSolverBase : public SolverBase<LinOp> {
public:
    /**
     * Gets the stopping criterion factory of the solver.
     *
     * @return the stopping criterion factory
     */
    std::shared_ptr<const stop::CriterionFactory> get_stop_criterion_factory()
        const;

    /**
     * Sets the stopping criterion of the solver.
     *
     * @param other  the new stopping criterion factory
     */
    void set_stop_criterion_factory(
        std::shared_ptr<const stop::CriterionFactory> new_stop_factory);

    IterativeSolverBase &operator=(const IterativeSolverBase &other);

    IterativeSolverBase &operator=(IterativeSolverBase &&other);

    IterativeSolverBase();

    IterativeSolverBase(
        std::shared_ptr<const LinOp> system_matrix,
        std::shared_ptr<const stop::CriterionFactory> stop_factory);

    IterativeSolverBase(const IterativeSolverBase &);

    IterativeSolverBase(IterativeSolverBase &&);

private:
    std::shared_ptr<const stop::CriterionFactory> stop_factory_;
};


/**
 * A LinOp implementing this interface stores a system matrix and stopping
 * criterion factory.
 *
 * @ingroup solver
 * @ingroup LinOp
 */
template <typename ValueType>
class PreconditionedIterativeSolverBase : public IterativeSolverBase,
                                          public Preconditionable {
public:
    PreconditionedIterativeSolverBase() {}

    PreconditionedIterativeSolverBase(
        std::shared_ptr<const LinOp> system_matrix,
        std::shared_ptr<const stop::CriterionFactory> stop_factory,
        std::shared_ptr<const LinOp> preconditioner)
        : IterativeSolverBase{std::move(system_matrix),
                              std::move(stop_factory)},
          Preconditionable{std::move(preconditioner)}
    {
        GKO_ASSERT_EQUAL_DIMENSIONS(this->get_preconditioner(),
                                    this->get_system_matrix());
        // TODO assert(system_matrix == nullptr || preconditioner == nullptr ||
        // system_matrix->get_executor() == preconditioner->get_executor());
        // ...
    }

    template <typename FactoryParameters>
    PreconditionedIterativeSolverBase(
        std::shared_ptr<const LinOp> system_matrix,
        const FactoryParameters &params)
        : PreconditionedIterativeSolverBase{
              system_matrix, stop::combine(params.criteria),
              generate_preconditioner(system_matrix, params)}
    {}

private:
    template <typename FactoryParameters>
    static std::shared_ptr<const LinOp> generate_preconditioner(
        std::shared_ptr<const LinOp> system_matrix,
        const FactoryParameters &params)
    {
        if (params.generated_preconditioner) {
            return params.generated_preconditioner;
        } else if (params.preconditioner) {
            return params.preconditioner->generate(system_matrix);
        } else {
            return matrix::Identity<ValueType>::create(
                system_matrix->get_executor(), system_matrix->get_size());
        }
    }
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_SOLVER_BASE_HPP_
