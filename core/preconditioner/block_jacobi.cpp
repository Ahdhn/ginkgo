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

#include "core/preconditioner/block_jacobi.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/matrix/dense.hpp"
#include "core/preconditioner/block_jacobi_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace {


template <typename... TArgs>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(generate, block_jacobi::generate<TArgs...>);
};


}  // namespace


template <typename ValueType, typename IndexType>
void BlockJacobi<ValueType, IndexType>::copy_from(const LinOp *other)
{
    // TODO
}


template <typename ValueType, typename IndexType>
void BlockJacobi<ValueType, IndexType>::copy_from(std::unique_ptr<LinOp> other)
{
    // TODO
}


template <typename ValueType, typename IndexType>
void BlockJacobi<ValueType, IndexType>::apply(const LinOp *b, LinOp *x) const
{
    // TODO
}


template <typename ValueType, typename IndexType>
void BlockJacobi<ValueType, IndexType>::apply(const LinOp *alpha,
                                              const LinOp *b, const LinOp *beta,
                                              LinOp *x) const
{
    // TODO
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> BlockJacobi<ValueType, IndexType>::clone_type() const
{
    // TODO
    return std::unique_ptr<LinOp>();
}


template <typename ValueType, typename IndexType>
void BlockJacobi<ValueType, IndexType>::clear()
{
    // TODO
}


template <typename ValueType, typename IndexType>
void BlockJacobi<ValueType, IndexType>::convert_to(
    BlockJacobi<ValueType, IndexType> *result) const
{
    // TODO
}


template <typename ValueType, typename IndexType>
void BlockJacobi<ValueType, IndexType>::move_to(
    BlockJacobi<ValueType, IndexType> *result)
{
    // TODO
}


template <typename ValueType, typename IndexType>
void BlockJacobi<ValueType, IndexType>::generate(const LinOp *system_matrix)
{
    // TODO
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> BlockJacobiFactory<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> base) const
{
    return BlockJacobi<ValueType, IndexType>::create(
        this->get_executor(), base.get(), max_block_size_, block_pointers_);
}


#define GKO_DECLARE_BLOCK_JACOBI(ValueType, IndexType) \
    class BlockJacobi<ValueType, IndexType>
#define GKO_DECLARE_BLOCK_JACOBI_FACTORY(ValueType, IndexType) \
    class BlockJacobiFactory<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BLOCK_JACOBI);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BLOCK_JACOBI_FACTORY);
#undef GKO_DECLARE_BLOCK_JACOBI
#undef GKO_DECLARE_BLOCK_JACOBI_FACTORY


}  // namespace preconditioner
}  // namespace gko
