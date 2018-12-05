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

#ifndef GKO_CORE_PRECONDITIONER_BLOCK_JACOBI_HPP_
#define GKO_CORE_PRECONDITIONER_BLOCK_JACOBI_HPP_


#include "core/base/array.hpp"
#include "core/base/lin_op.hpp"
#include "core/matrix/dense.hpp"


namespace gko {


/**
 * This type is used to store data about precision of diagonal blocks.
 */
enum precision {
    /**
     * Marks that double precision is used for the block.
     */
    double_precision,

    /**
     * Marks that single precision is used for the block.
     */
    single_precision,

    /**
     * Marks that half precision is used for the block.
     */
    half_precision,

    /**
     * The precision of the block will be determined automatically during
     * generation phase, based on the condition number of the block.
     */
    best_precision
};


namespace preconditioner {


// TODO: replace this with a custom accessor
/**
 * Defines the parameters of the interleaved block storage scheme used by
 * block-Jacobi blocks.
 *
 * @tparam IndexType  type used for storing indices of the matrix
 */
template <typename IndexType>
struct block_interleaved_storage_scheme {
    /**
     * The offset between consecutive blocks within the group.
     */
    IndexType block_offset;
    /**
     * The offset between two block groups.
     */
    IndexType group_offset;
    /**
     * Then base 2 power of the group.
     *
     * I.e. the group contains `1 << group_power` elements.
     */
    uint32 group_power;

    /**
     * Returns the number of elements in the group.
     *
     * @return the number of elements in the group
     */
    GKO_ATTRIBUTES IndexType get_group_size() const noexcept
    {
        return one<IndexType>() << group_power;
    }

    /**
     * Computes the storage space required for the requested number of blocks.
     *
     * @param num_blocks  the total number of blocks that needs to be stored
     *
     * @return the total memory (as the number of elements) that need to be
     *         allocated for the scheme
     */
    GKO_ATTRIBUTES IndexType compute_storage_space(IndexType num_blocks) const
        noexcept
    {
        return (num_blocks + 1 == size_type{0})
                   ? size_type{0}
                   : ceildiv(num_blocks, this->get_group_size()) * group_offset;
    }

    /**
     * Returns the offset of the group belonging to the block with the given ID.
     *
     * @param block_id  the ID of the block
     *
     * @return the offset of the group belonging to block with ID `block_id`
     */
    GKO_ATTRIBUTES IndexType get_group_offset(IndexType block_id) const noexcept
    {
        return group_offset * (block_id >> group_power);
    }

    /**
     * Returns the offset of the block with the given ID within its group.
     *
     * @param block_id  the ID of the block
     *
     * @return the offset of the block with ID `block_id` within its group
     */
    GKO_ATTRIBUTES IndexType get_block_offset(IndexType block_id) const noexcept
    {
        return block_offset * (block_id & (this->get_group_size() - 1));
    }

    /**
     * Returns the offset of the block with the given ID.
     *
     * @param block_id  the ID of the block
     *
     * @return the offset of the block with ID `block_id`
     */
    GKO_ATTRIBUTES IndexType get_global_block_offset(IndexType block_id) const
        noexcept
    {
        return this->get_group_offset(block_id) +
               this->get_block_offset(block_id);
    }

    /**
     * Returns the stride between columns of the block.
     *
     * @return stride between columns of the block
     */
    GKO_ATTRIBUTES IndexType get_stride() const noexcept
    {
        return block_offset << group_power;
    }
};


/**
 * A block-Jacobi preconditioner is a block-diagonal linear operator, obtained
 * by inverting the diagonal blocks of another operator.
 *
 * The Jacobi class implements the inversion of the diagonal blocks using
 * Gauss-Jordan elimination with column pivoting, and stores the inverse
 * explicitly in a customized format.
 *
 * If the diagonal blocks of the matrix are not explicitly set by the user, the
 * implementation will try to automatically detect the blocks by first finding
 * the natural blocks of the matrix, and then applying the supervariable
 * agglomeration procedure on them.
 *
 * This class also implements an improved adaptive version of the block-Jacobi
 * preconditioner, which can store some of the blocks in lower precision and
 * thus improve the performance of preconditioner application by reducing the
 * amount of memory that has to be read to apply the preconditioner.
 * This can be enabled by setting the `global_precision` (or, for advanced uses
 * `block_precisions`) parameter. Refer to the documentation of these parameters
 * for more details.
 * However, there is a trade-off in terms of slightly longer preconditioner
 * generation when using the adaptive variant due to extra work required to
 * compute the condition numbers (a step is necessary to preserve the
 * regularity of the diagonal blocks).
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  integral type used to store pointers to the start of each
 *                    block
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Jacobi : public EnableLinOp<Jacobi<ValueType, IndexType>>,
               public ConvertibleTo<matrix::Dense<ValueType>>,
               public WritableToMatrixData<ValueType, IndexType> {
    friend class EnableLinOp<Jacobi>;
    friend class EnablePolymorphicObject<Jacobi, LinOp>;

public:
    using EnableLinOp<Jacobi>::convert_to;
    using EnableLinOp<Jacobi>::move_to;
    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;

    /**
     * Returns the number of blocks of the operator.
     *
     * @return the number of blocks of the operator
     *
     * TODO: replace with ranges
     */
    size_type get_num_blocks() const noexcept { return num_blocks_; }

    /**
     * Returns the storage scheme used for storing Jacobi blocks.
     *
     * @return the storage scheme used for storing Jacobi blocks
     *
     * TODO: replace with ranges
     */
    const block_interleaved_storage_scheme<index_type> &get_storage_scheme()
        const noexcept
    {
        return storage_scheme_;
    }

    /**
     * Returns the pointer to the memory used for storing the block data.
     *
     * Element (`i`, `j`) of block `b` is stored in position
     * `(get_block_pointers()[b] + i) * stride + j` of the array.
     *
     * @return the pointer to the memory used for storing the block data
     *
     * TODO: replace with ranges
     */
    const value_type *get_blocks() const noexcept
    {
        return blocks_.get_const_data();
    }

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_stored_elements() const noexcept
    {
        return blocks_.get_num_elems();
    }

    void convert_to(matrix::Dense<value_type> *result) const override;

    void move_to(matrix::Dense<value_type> *result) override;

    void write(mat_data &data) const override;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Maximal size of diagonal blocks.
         *
         * @note This values has to be between 1 and 32.
         */
        uint32 GKO_FACTORY_PARAMETER(max_block_size, 32u);

        /**
         * Starting (row / column) index of each individual block.
         *
         * An index past the last block has to be supplied as the last value.
         * I.e. the size of the array has to be the number of blocks plus 1,
         * where the first value is 0, and the last value is the number of
         * rows / columns of the matrix.
         *
         * @note Even if not set explicitly, this parameter will be set to
         *       automatically detected values once the preconditioner is
         *       generated.
         * @note If the parameter is set automatically, the size of the array
         *       does not correlate to the number of blocks, and is
         *       implementation defined. To obtain the number of blocks `n` use
         *       Jacobi::get_num_blocks(). The starting indexes of each block
         *       are stored in the first `n+1` values of this array.
         */
        gko::Array<index_type> GKO_FACTORY_PARAMETER(block_pointers);

        /**
         * Global precision to use for all blocks.
         *
         * This parameter only has effect if block_precisions is not set.
         *
         * If the value is set to `double_precision`, standard block-Jacobi
         * preconditioner will be used.
         *
         * If the value is set to `best_precision`, the adaptive version will be
         * used. The adaptive version automatically chooses the precision of
         * each individual block, depending on its condition number.
         *
         * If the value is set to `single_precision` or `half_precision`, the
         * mixed precision version is used. The mixed precision version will
         * store all blocks using the same (but lower, depending on the
         * variable) precision.
         *
         * If better control is needed, the precision of each block can be set
         * individually by using the block_precisions parameter.
         */
        precision GKO_FACTORY_PARAMETER(global_precision, double_precision);

        /**
         * Precisions to use for each individual block.
         *
         * If left unset and global_precision is set to double_precision, the
         * standard full-precision version will be used. Otherwise,
         * adaptive/mixed precision version is used.
         *
         * If the number of blocks is larger than the number of elements in this
         * array, the original array will be replicated multiple times, until
         * its length reaches the number of blocks of the preconditioner.
         *
         * Each value in the list specifies the precision of the corresponding
         * block and has the same semantics as in the global_precision
         * parameter. The precisions of blocks whose corresponding entries in
         * the array are set to `best_precision` will be detected automatically.
         *
         * @internal
         *
         * @note Once Jacobi's constructor has been called, this array will be
         *       empty if and only if the non-adaptive version of Jacobi has
         *       been requested.
         */
        gko::Array<precision> GKO_FACTORY_PARAMETER(block_precisions);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Jacobi, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit Jacobi(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Jacobi>(exec), blocks_(exec)
    {
        parameters_.block_pointers.set_executor(exec);
        parameters_.block_precisions.set_executor(exec);
    }

    explicit Jacobi(const Factory *factory,
                    std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Jacobi>(factory->get_executor(),
                              transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          storage_scheme_{compute_storage_scheme(parameters_.max_block_size)},
          num_blocks_{parameters_.block_pointers.get_num_elems() - 1},
          blocks_(factory->get_executor(),
                  storage_scheme_.compute_storage_space(
                      parameters_.block_pointers.get_num_elems() - 1))
    {
        if (parameters_.max_block_size >= 32 ||
            parameters_.max_block_size < 1) {
            NOT_SUPPORTED(this);
        }
        parameters_.block_pointers.set_executor(this->get_executor());
        if (parameters_.block_precisions.get_num_elems() == 0 &&
            parameters_.global_precision != double_precision) {
            parameters_.block_precisions = gko::Array<precision>(
                this->get_executor(), {parameters_.global_precision});
        } else {
            parameters_.block_precisions.set_executor(this->get_executor());
        }
        this->generate(lend(system_matrix));
    }

    /**
     * Stride between two columns of a block (as number of elements).
     *
     * Should be a multiple of cache line size for best performance.
     */
    static constexpr size_type max_block_stride_ = 32;

    /**
     * Returns the smallest power of 2 at least as larger as the input.
     *
     * @param n  a number
     *
     * @return a power of two at least as large as `n`
     */
    static size_type get_larger_power(size_type n) noexcept
    {
        size_type res = 1;
        while (res < n) res *= 2;
        return res;
    }

    /**
     * Returns the base-2 logarithm of `n` rounded down to the nearest integer.
     */
    static uint32 get_log2(size_type n) noexcept
    {
        for (auto r = uint32{0};; ++r) {
            if ((size_type{1} << (r + 1)) > n) {
                return r;
            }
        }
    }

    /**
     * Computes the storage scheme suitable for storing blocks of a given
     * maximum size.
     *
     * @param max_block_size  the maximum size of the blocks
     *
     * @return a suitable storage scheme
     */
    static block_interleaved_storage_scheme<index_type> compute_storage_scheme(
        uint32 max_block_size) noexcept
    {
        const auto group_size = static_cast<uint32>(
            max_block_stride_ / get_larger_power(max_block_size));
        const auto block_offset = max_block_size;
        const auto block_stride = group_size * block_offset;
        const auto group_offset = max_block_size * block_stride;
        return {static_cast<index_type>(block_offset),
                static_cast<index_type>(group_offset), get_log2(group_size)};
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    /**
     * Generates the preconditoner.
     *
     * @param system_matrix  the source matrix used to generate the
     *                       preconditioner
     */
    void generate(const LinOp *system_matrix);

    /**
     * Detects the diagonal blocks and allocates the memory needed to store the
     * preconditioner.
     *
     * @param system_matrix  the source matrix whose diagonal block patter is to
     *                       be detected
     */
    void detect_blocks(const matrix::Csr<ValueType, IndexType> *system_matrix);

private:
    block_interleaved_storage_scheme<index_type> storage_scheme_{};
    size_type num_blocks_;
    Array<value_type> blocks_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_BLOCK_JACOBI_HPP_
