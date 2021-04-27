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

#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {


std::shared_ptr<const LinOp> Preconditionable::get_preconditioner() const
{
    return preconditioner_;
}


void Preconditionable::set_preconditioner(
    std::shared_ptr<const LinOp> new_precond)
{
    if (preconditioner_ == nullptr || new_precond == nullptr ||
        preconditioner_->get_executor() == new_precond->get_executor()) {
        preconditioner_ = std::move(new_precond);
    } else {
        preconditioner_ = clone(preconditioner_->get_executor(), new_precond);
    }
}


Preconditionable &Preconditionable::operator=(const Preconditionable &other)
{
    this->set_preconditioner(other.get_preconditioner());
    return *this;
}


Preconditionable &Preconditionable::operator=(Preconditionable &&other)
{
    this->set_preconditioner(other.get_preconditioner());
    other.set_preconditioner(nullptr);
    return *this;
}


Preconditionable::Preconditionable() : Preconditionable{nullptr} {}


Preconditionable::Preconditionable(std::shared_ptr<const LinOp> preconditioner)
    : preconditioner_{std::move(preconditioner)}
{}


Preconditionable::Preconditionable(const Preconditionable &other)
    : Preconditionable{}
{
    *this = other;
}


Preconditionable::Preconditionable(Preconditionable &&other)
    : Preconditionable{}
{
    *this = std::move(other);
}


namespace solver {


std::shared_ptr<const stop::CriterionFactory>
IterativeSolverBase::get_stop_criterion_factory() const
{
    return stop_factory_;
}


void IterativeSolverBase::set_stop_criterion_factory(
    std::shared_ptr<const stop::CriterionFactory> new_stop_factory)
{
    if (stop_factory_ == nullptr || new_stop_factory == nullptr ||
        stop_factory_->get_executor() == new_stop_factory->get_executor()) {
        stop_factory_ = std::move(new_stop_factory);
    } else {
        stop_factory_ = clone(stop_factory_->get_executor(), new_stop_factory);
    }
}


IterativeSolverBase &IterativeSolverBase::operator=(
    const IterativeSolverBase &other)
{
    this->SolverBase::operator=(other);
    this->set_stop_criterion_factory(other.get_stop_criterion_factory());
    return *this;
}


IterativeSolverBase &IterativeSolverBase::operator=(IterativeSolverBase &&other)
{
    this->SolverBase::operator=(std::move(other));
    this->set_stop_criterion_factory(other.get_stop_criterion_factory());
    other.set_stop_criterion_factory(nullptr);
    return *this;
}


IterativeSolverBase::IterativeSolverBase() {}


IterativeSolverBase::IterativeSolverBase(
    std::shared_ptr<const LinOp> system_matrix,
    std::shared_ptr<const stop::CriterionFactory> stop_factory)
    : SolverBase{std::move(system_matrix)},
      stop_factory_{std::move(stop_factory)}
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this->get_system_matrix());
    // TODO assert(system_matrix == nullptr || stop_factory == nullptr ||
    // system_matrix->get_executor() == stop_factory->get_executor());
    // ...
}


IterativeSolverBase::IterativeSolverBase(const IterativeSolverBase &other)
    : IterativeSolverBase{}
{
    *this = other;
}


IterativeSolverBase::IterativeSolverBase(IterativeSolverBase &&other)
    : IterativeSolverBase{}
{
    *this = std::move(other);
}


}  // namespace solver
}  // namespace gko
