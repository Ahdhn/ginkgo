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

#ifndef GKO_CORE_SOLVER_XXSOLVERXX_HPP_
#define GKO_CORE_SOLVER_XXSOLVERXX_HPP_


#include "core/base/array.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"
#include "core/matrix/identity.hpp"


namespace gko {
namespace solver {


template <typename>
class XxsolverxxFactory;

/**
 * BiCGSTAB or the Bi-Conjugate Gradient-Stabilized is a Krylov subspace solver.
 *
 * Being a generic solver, it is capable of solving general matrices, including
 * non-s.p.d matrices. Though, the memory and the computational requirement of
 * the BiCGSTAB solver are higher than of its s.p.d solver counterpart, it has
 * the capability to solve generic systems. It was developed by stabilizing the
 * BiCG method.
 *
 * @tparam ValueType precision of the elements of the system matrix.
 */
template <typename ValueType = default_precision>
class Xxsolverxx : public BasicLinOp<Xxsolverxx<ValueType>>,
                   public PreconditionedMethod {
    friend class BasicLinOp<Xxsolverxx>;
    friend class XxsolverxxFactory<ValueType>;

public:
    using BasicLinOp<Xxsolverxx>::convert_to;
    using BasicLinOp<Xxsolverxx>::move_to;

    using value_type = ValueType;

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
               LinOp *x) const override;

    /**
     * Gets the system matrix of the linear system.
     *
     * @return  The system matrix.
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    /**
     * Gets the maximum number of iterations of the BiCGSTAB solver.
     *
     * @return  The maximum number of iterations.
     */
    int get_max_iters() const { return max_iters_; }

    /**
     * Gets the relative residual goal of the solver.
     *
     * @return  The relative residual goal.
     */
    remove_complex<value_type> get_rel_residual_goal() const
    {
        return rel_residual_goal_;
    }

protected:
    using BasicLinOp<Xxsolverxx>::create;

    explicit Xxsolverxx(std::shared_ptr<const Executor> exec)
        : BasicLinOp<Xxsolverxx>(exec, 0, 0, 0)
    {}

    Xxsolverxx(std::shared_ptr<const Executor> exec, int max_iters,
               remove_complex<value_type> rel_residual_goal,
               std::shared_ptr<const LinOp> system_matrix)
        : BasicLinOp<Xxsolverxx>(
              exec, system_matrix->get_num_cols(),
              system_matrix->get_num_rows(),
              system_matrix->get_num_rows() * system_matrix->get_num_cols()),
          system_matrix_(std::move(system_matrix)),
          max_iters_(max_iters),
          rel_residual_goal_(rel_residual_goal)
    {}

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    int max_iters_{};
    remove_complex<value_type> rel_residual_goal_{};
};


/**
 * Creates the BiCGSTAB solver.
 *
 * @param exec The executor on which the BiCGSTAB solver is to be created.
 * @param max_iters  The maximum number of iterations to be pursued.
 * @param rel_residual_goal  The relative residual required for
 * convergence.
 *
 * @return The newly created BiCGSTAB solver.
 */
template <typename ValueType = default_precision>
class XxsolverxxFactory : public LinOpFactory,
                          public PreconditionedMethodFactory {
public:
    using value_type = ValueType;

    static std::unique_ptr<XxsolverxxFactory> create(
        std::shared_ptr<const Executor> exec, int max_iters,
        remove_complex<value_type> rel_residual_goal)
    {
        return std::unique_ptr<XxsolverxxFactory>(new XxsolverxxFactory(
            std::move(exec), max_iters, rel_residual_goal));
    }

    std::unique_ptr<LinOp> generate(
        std::shared_ptr<const LinOp> base) const override;

    /**
     * Gets the maximum number of iterations of the BiCGSTAB solver.
     *
     * @return  The maximum number of iterations.
     */
    int get_max_iters() const { return max_iters_; }

    /**
     * Gets the relative residual goal of the solver.
     *
     * @return  The relative residual goal.
     */
    remove_complex<value_type> get_rel_residual_goal() const
    {
        return rel_residual_goal_;
    }

protected:
    XxsolverxxFactory(std::shared_ptr<const Executor> exec, int max_iters,
                      remove_complex<value_type> rel_residual_goal)
        : LinOpFactory(exec),
          PreconditionedMethodFactory(
              matrix::IdentityFactory<ValueType>::create(std::move(exec))),
          max_iters_(max_iters),
          rel_residual_goal_(rel_residual_goal)
    {}

    int max_iters_;
    remove_complex<value_type> rel_residual_goal_;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_CORE_SOLVER_XXSOLVERXX_HPP
