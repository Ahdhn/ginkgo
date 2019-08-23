/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_FACTORIZATION_QR_HPP_
#define GKO_CORE_FACTORIZATION_QR_HPP_


#include <memory>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
/**
 * @brief The Factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


/**
 * QR is reduced QR factorization. The implementation is the Householder QR
 * factorization without pivoting.
 *
 * $A = Q \times R$
 * $Q$ is an orthogonal matrix and $R$ is a upper triangular matrix.
 * $Q$ is not formed explicitly. $Q$ is composition of a series of Householder
 * matrix.
 *
 * The Housholder QR facorization follows the design of  Trefethen, Lloyd N.,
 * and David Bau III. Numerical linear algebra. Vol. 50. Siam, 1997.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup factor
 * @ingroup linop
 */
template <typename ValueType = default_precision>
class Qr : public Composition<ValueType> {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using q_matrix_type = Composition<ValueType>;
    using r_matrix_type = Dense<ValueType>;

    const q_matrix_type *get_q_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return static_cast<const q_type *>(this->get_operators()[0].get());
    }

    const r_matirx_type *get_r_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return static_cast<const r_matrix_type *>(
            this->get_operators()[1].get());
    }

    // Remove the possibility of calling `create`, which was enabled by
    // `Composition`
    template <typename... Args>
    static std::unique_ptr<Composition<ValueType>> create(Args &&... args) =
        delete;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * It performs the rank QR factorization. The default value `0` means
         * `full rank`, so the implementation compuete the full rank QR
         * factorization.
         */
        size_type GKO_FACTORY_PARAMETER(rank, 0);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Qr, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit Qr(const Factory *factory,
                std::shared_ptr<const LinOp> system_matrix)
        : Composition<ValueType>(factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        validate_qr(system_matrix);
        generate_qr(system_matrix)->move_to(this);
    }


    /**
     * Validates the dimensions of the matrix. The number of columns cannot be
     * larger than the number of rows and the rank cannot be larger than the
     * number of columns.
     */
    void validate_qr(const std::shared_ptr<const LinOp> system_matrix)
    {
        const auto matrix_size = system_matrix.get_size();
        if (parameters_.rank > matrix_size[1] ||
            matrix_size[1] > matrix_size[0]) {
            GKO_NOT_SUPPORTED(this);
        }
    }
    /**
     * Generates the QR factors, which will be returned as a
     * composition of the orthogonal matrix (first element of the composition)
     * and the upper triangular matrix (second element)
     *
     * @param system_matrix  the source matrix used to generate the factors.
     *                       @note: system_matrix must be convertable to a Dense
     *                              Matrix, otherwise, an exception is thrown.
     * @return  A Composition, containing the QR factors for the
     *          given system_matrix (first element is Q, then R)
     */
    std::unique_ptr<Composition<ValueType>> generate_qr(
        const std::shared_ptr<const LinOp> &system_matrix) const;
};


}  // namespace factorization
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_QR_HPP_
