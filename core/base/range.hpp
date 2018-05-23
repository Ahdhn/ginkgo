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

#ifndef GKO_CORE_BASE_RANGE_HPP_
#define GKO_CORE_BASE_RANGE_HPP_


#include "core/base/math.hpp"
#include "core/base/std_extensions.hpp"
#include "core/base/types.hpp"
#include "core/base/utils.hpp"


namespace gko {


struct span {
    GKO_ATTRIBUTES constexpr span(size_type point) noexcept
        : span{point, point + 1}
    {}

    GKO_ATTRIBUTES constexpr span(size_type begin, size_type end) noexcept
        : begin{begin}, end{end}
    {}

    GKO_ATTRIBUTES static constexpr span empty(size_type point) noexcept
    {
        return {point, point};
    }

    const size_type begin;
    const size_type end;
};


GKO_ATTRIBUTES GKO_INLINE constexpr bool operator<(const span &first,
                                                   const span &second)
{
    return first.end < second.begin;
}


GKO_ATTRIBUTES GKO_INLINE constexpr bool operator<=(const span &first,
                                                    const span &second)
{
    return first.end <= second.begin;
}


GKO_ATTRIBUTES GKO_INLINE constexpr bool operator>(const span &first,
                                                   const span &second)
{
    return second < first;
}


GKO_ATTRIBUTES GKO_INLINE constexpr bool operator>=(const span &first,
                                                    const span &second)
{
    return second <= first;
}


GKO_ATTRIBUTES GKO_INLINE constexpr bool operator==(const span &first,
                                                    const span &second)
{
    return first.begin == second.begin && first.end == second.end;
}


GKO_ATTRIBUTES GKO_INLINE constexpr bool operator!=(const span &first,
                                                    const span &second)
{
    return !(first == second);
}


namespace detail {


template <size_type CurrentDimension, size_type Dimensionality,
          typename FirstRange, typename SecondRange>
GKO_ATTRIBUTES GKO_INLINE
    xstd::enable_if_t<(CurrentDimension >= Dimensionality), bool>
    equal_dimensions_impl(const FirstRange &, const SecondRange &)
{
    return true;
}

template <size_type CurrentDimension, size_type Dimensionality,
          typename FirstRange, typename SecondRange>
GKO_ATTRIBUTES GKO_INLINE
    xstd::enable_if_t<(CurrentDimension < Dimensionality), bool>
    equal_dimensions_impl(const FirstRange &first, const SecondRange &second)
{
    return first.length(CurrentDimension) == second.length(CurrentDimension) &&
           equal_dimensions_impl<CurrentDimension + 1, Dimensionality>(first,
                                                                       second);
}

template <typename FirstRange, typename SecondRange>
GKO_ATTRIBUTES GKO_INLINE bool equal_dimensions(const FirstRange &first,
                                                const SecondRange &second)
{
    return FirstRange::dimensionality == SecondRange::dimensionality &&
           equal_dimensions_impl<0, FirstRange::dimensionality>(first, second);
}


}  // namespace detail


template <typename Accessor>
class range {
public:
    using accessor = Accessor;

    static constexpr size_type dimensionality = accessor::dimensionality;

    template <typename... AccessorParams>
    GKO_ATTRIBUTES explicit range(AccessorParams &&... params)
        : accessor_{std::forward<AccessorParams>(params)...}
    {}

    template <typename... DimensionsType>
    GKO_ATTRIBUTES auto operator()(DimensionsType &&... dimensions) const
        -> decltype(std::declval<accessor>()(
            std::forward<DimensionsType>(dimensions)...))
    {
        static_assert(sizeof...(dimensions) <= dimensionality,
                      "Too many dimensions in range call");
        return accessor_(std::forward<DimensionsType>(dimensions)...);
    }

    template <typename OtherAccessor>
    GKO_ATTRIBUTES const range &operator=(
        const range<OtherAccessor> &other) const
    {
        GKO_ASSERT(detail::equal_dimensions(*this, other));
        accessor_.copy_from(other);
        return *this;
    }

    GKO_ATTRIBUTES const range &operator=(const range &other) const
    {
        GKO_ASSERT(detail::equal_dimensions(*this, other));
        accessor_.copy_from(other.get_accessor());
        return *this;
    }

    GKO_ATTRIBUTES size_type length(size_type dimension) const
    {
        return accessor_.length(dimension);
    }

    const accessor *operator->() const noexcept { return &accessor_; }

    GKO_ATTRIBUTES const accessor &get_accessor() const noexcept
    {
        return accessor_;
    }

private:
    accessor accessor_;
};


// implementation of range operations


namespace detail {


enum class operation_kind { range_by_range, scalar_by_range, range_by_scalar };


template <typename Accessor, typename Operation>
struct implement_unary_operation {
    using accessor = Accessor;
    static constexpr size_type dimensionality = accessor::dimensionality;

    GKO_ATTRIBUTES explicit implement_unary_operation(const Accessor &operand)
        : operand{operand}
    {}

    template <typename... DimensionTypes>
    GKO_ATTRIBUTES auto operator()(const DimensionTypes &... dimensions) const
        -> decltype(Operation::evaluate(std::declval<accessor>(),
                                        dimensions...))
    {
        return Operation::evaluate(operand, dimensions...);
    }

    GKO_ATTRIBUTES size_type length(size_type dimension) const
    {
        return operand.length(dimension);
    }

    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const = delete;

    const accessor operand;
};


template <operation_kind Kind, typename FirstOperand, typename SecondOperand,
          typename Operation>
struct implement_binary_operation {};

template <typename FirstAccessor, typename SecondAccessor, typename Operation>
struct implement_binary_operation<operation_kind::range_by_range, FirstAccessor,
                                  SecondAccessor, Operation> {
    using first_accessor = FirstAccessor;
    using second_accessor = SecondAccessor;
    static_assert(first_accessor::dimensionality ==
                      second_accessor::dimensionality,
                  "Both ranges need to have the same number of dimensions");
    static constexpr size_type dimensionality = first_accessor::dimensionality;

    GKO_ATTRIBUTES explicit implement_binary_operation(
        const FirstAccessor &first, const SecondAccessor &second)
        : first{first}, second{second}
    {
        GKO_ASSERT(gko::detail::equal_dimensions(first, second));
    }

    template <typename... DimensionTypes>
    GKO_ATTRIBUTES auto operator()(const DimensionTypes &... dimensions) const
        -> decltype(Operation::evaluate_range_by_range(
            std::declval<first_accessor>(), std::declval<second_accessor>(),
            dimensions...))
    {
        return Operation::evaluate_range_by_range(first, second, dimensions...);
    }

    GKO_ATTRIBUTES size_type length(size_type dimension) const
    {
        return first.length(dimension);
    }

    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const = delete;

    const first_accessor first;
    const second_accessor second;
};

template <typename FirstOperand, typename SecondAccessor, typename Operation>
struct implement_binary_operation<operation_kind::scalar_by_range, FirstOperand,
                                  SecondAccessor, Operation> {
    using second_accessor = SecondAccessor;
    static constexpr size_type dimensionality = second_accessor::dimensionality;

    GKO_ATTRIBUTES explicit implement_binary_operation(
        const FirstOperand &first, const SecondAccessor &second)
        : first{first}, second{second}
    {}

    template <typename... DimensionTypes>
    GKO_ATTRIBUTES auto operator()(const DimensionTypes &... dimensions) const
        -> decltype(Operation::evaluate_scalar_by_range(
            std::declval<FirstOperand>(), std::declval<second_accessor>(),
            dimensions...))
    {
        return Operation::evaluate_scalar_by_range(first, second,
                                                   dimensions...);
    }

    GKO_ATTRIBUTES size_type length(size_type dimension) const
    {
        return second.length(dimension);
    }

    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const = delete;

    const FirstOperand first;
    const second_accessor second;
};

template <typename FirstAccessor, typename SecondOperand, typename Operation>
struct implement_binary_operation<operation_kind::range_by_scalar,
                                  FirstAccessor, SecondOperand, Operation> {
    using first_accessor = FirstAccessor;
    static constexpr size_type dimensionality = first_accessor::dimensionality;

    GKO_ATTRIBUTES explicit implement_binary_operation(
        const FirstAccessor &first, const SecondOperand &second)
        : first{first}, second{second}
    {}

    template <typename... DimensionTypes>
    GKO_ATTRIBUTES auto operator()(const DimensionTypes &... dimensions) const
        -> decltype(Operation::evaluate_range_by_scalar(
            std::declval<first_accessor>(), std::declval<SecondOperand>(),
            dimensions...))
    {
        return Operation::evaluate_range_by_scalar(first, second,
                                                   dimensions...);
    }

    GKO_ATTRIBUTES size_type length(size_type dimension) const
    {
        return first.length(dimension);
    }

    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const = delete;

    const first_accessor first;
    const SecondOperand second;
};


}  // namespace detail


#define GKO_ENABLE_UNARY_RANGE_OPERATION(_operation_name, _operator_name, \
                                         _operator)                       \
    namespace accessor {                                                  \
    template <typename Operand>                                           \
    struct _operation_name                                                \
        : ::gko::detail::implement_unary_operation<Operand,               \
                                                   ::gko::_operator> {    \
        using ::gko::detail::implement_unary_operation<                   \
            Operand, ::gko::_operator>::implement_unary_operation;        \
    };                                                                    \
    }                                                                     \
                                                                          \
    template <typename Accessor>                                          \
    GKO_ATTRIBUTES GKO_INLINE range<accessor::_operation_name<Accessor>>  \
    _operator_name(const range<Accessor> &operand)                        \
    {                                                                     \
        return range<accessor::_operation_name<Accessor>>(                \
            operand.get_accessor());                                      \
    }


#define GKO_DEFINE_SIMPLE_UNARY_OPERATION(_name, ...)                  \
    struct _name {                                                     \
    private:                                                           \
        template <typename Operand>                                    \
        GKO_ATTRIBUTES static auto simple_evaluate_impl(               \
            const Operand &operand) -> decltype(__VA_ARGS__)           \
        {                                                              \
            return __VA_ARGS__;                                        \
        }                                                              \
                                                                       \
    public:                                                            \
        template <typename AccessorType, typename... DimensionTypes>   \
        GKO_ATTRIBUTES static auto evaluate(                           \
            const AccessorType &accessor,                              \
            const DimensionTypes &... dimensions)                      \
            -> decltype(simple_evaluate_impl(accessor(dimensions...))) \
        {                                                              \
            return simple_evaluate_impl(accessor(dimensions...));      \
        }                                                              \
    };


namespace accessor {
namespace detail {


// unary arithmetic
GKO_DEFINE_SIMPLE_UNARY_OPERATION(unary_plus, +operand);
GKO_DEFINE_SIMPLE_UNARY_OPERATION(unary_minus, -operand);

// unary logical
GKO_DEFINE_SIMPLE_UNARY_OPERATION(logical_not, !operand);

// unary bitwise
GKO_DEFINE_SIMPLE_UNARY_OPERATION(bitwise_not, ~operand);

// common functions
GKO_DEFINE_SIMPLE_UNARY_OPERATION(zero_operation, zero(operand));
GKO_DEFINE_SIMPLE_UNARY_OPERATION(one_operation, one(operand));
GKO_DEFINE_SIMPLE_UNARY_OPERATION(abs_operation, abs(operand));
GKO_DEFINE_SIMPLE_UNARY_OPERATION(real_operation, real(operand));
GKO_DEFINE_SIMPLE_UNARY_OPERATION(imag_operation, imag(operand));
GKO_DEFINE_SIMPLE_UNARY_OPERATION(conj_operation, conj(operand));
GKO_DEFINE_SIMPLE_UNARY_OPERATION(squared_norm_operation,
                                  squared_norm(operand));

// special range functions
struct transpose_operation {
    template <typename AccessorType, typename FirstDimensionType,
              typename SecondDimensionType, typename... DimensionTypes>
    GKO_ATTRIBUTES static auto evaluate(const AccessorType &accessor,
                                        const FirstDimensionType &first_dim,
                                        const SecondDimensionType &second_dim,
                                        const DimensionTypes &... dims)
        -> decltype(accessor(second_dim, first_dim, dims...))
    {
        return accessor(second_dim, first_dim, dims...);
    }
};


}  // namespace detail
}  // namespace accessor


// unary arithmetic
GKO_ENABLE_UNARY_RANGE_OPERATION(unary_plus, operator+,
                                 accessor::detail::unary_plus);
GKO_ENABLE_UNARY_RANGE_OPERATION(unary_minus, operator-,
                                 accessor::detail::unary_minus);

// unary logical
GKO_ENABLE_UNARY_RANGE_OPERATION(logical_not, operator!,
                                 accessor::detail::logical_not);

// unary bitwise
GKO_ENABLE_UNARY_RANGE_OPERATION(bitwise_not, operator~,
                                 accessor::detail::bitwise_not);

// common unary functions
GKO_ENABLE_UNARY_RANGE_OPERATION(zero_operation, zero,
                                 accessor::detail::zero_operation);
GKO_ENABLE_UNARY_RANGE_OPERATION(one_operaton, one,
                                 accessor::detail::one_operation);
GKO_ENABLE_UNARY_RANGE_OPERATION(abs_operaton, abs,
                                 accessor::detail::abs_operation);
GKO_ENABLE_UNARY_RANGE_OPERATION(real_operaton, real,
                                 accessor::detail::real_operation);
GKO_ENABLE_UNARY_RANGE_OPERATION(imag_operaton, imag,
                                 accessor::detail::imag_operation);
GKO_ENABLE_UNARY_RANGE_OPERATION(conj_operaton, conj,
                                 accessor::detail::conj_operation);
GKO_ENABLE_UNARY_RANGE_OPERATION(squared_norm_operaton, squared_norm,
                                 accessor::detail::squared_norm_operation);

// special range functions

/**
 * Transposes the first two dimensions of the input range.
 *
 * Thus, the following equality holds for any range:
 *     transpose(range)(i, j, ...) == range(j, i, ....).
 */
GKO_ENABLE_UNARY_RANGE_OPERATION(transpose_operaton, transpose,
                                 accessor::detail::transpose_operation);


#undef GKO_DEFINE_SIMPLE_UNARY_OPERATION
#undef GKO_ENABLE_UNARY_RANGE_OPERATION


#define GKO_ENABLE_BINARY_RANGE_OPERATION(_operation_name, _operator_name,  \
                                          _operator)                        \
    namespace accessor {                                                    \
    template <::gko::detail::operation_kind Kind, typename FirstOperand,    \
              typename SecondOperand>                                       \
    struct _operation_name                                                  \
        : ::gko::detail::implement_binary_operation<                        \
              Kind, FirstOperand, SecondOperand, ::gko::_operator> {        \
        using ::gko::detail::implement_binary_operation<                    \
            Kind, FirstOperand, SecondOperand,                              \
            ::gko::_operator>::implement_binary_operation;                  \
    };                                                                      \
    }                                                                       \
                                                                            \
    template <typename Accessor>                                            \
    GKO_ATTRIBUTES GKO_INLINE range<accessor::_operation_name<              \
        ::gko::detail::operation_kind::range_by_range, Accessor, Accessor>> \
    _operator_name(const range<Accessor> &first,                            \
                   const range<Accessor> &second)                           \
    {                                                                       \
        return range<accessor::_operation_name<                             \
            ::gko::detail::operation_kind::range_by_range, Accessor,        \
            Accessor>>(first.get_accessor(), second.get_accessor());        \
    }                                                                       \
                                                                            \
    template <typename FirstAccessor, typename SecondAccessor>              \
    GKO_ATTRIBUTES GKO_INLINE range<accessor::_operation_name<              \
        ::gko::detail::operation_kind::range_by_range, FirstAccessor,       \
        SecondAccessor>>                                                    \
    _operator_name(const range<FirstAccessor> &first,                       \
                   const range<SecondAccessor> &second)                     \
    {                                                                       \
        return range<accessor::_operation_name<                             \
            ::gko::detail::operation_kind::range_by_range, FirstAccessor,   \
            SecondAccessor>>(first.get_accessor(), second.get_accessor());  \
    }                                                                       \
                                                                            \
    template <typename FirstAccessor, typename SecondOperand>               \
    GKO_ATTRIBUTES GKO_INLINE range<accessor::_operation_name<              \
        ::gko::detail::operation_kind::range_by_scalar, FirstAccessor,      \
        SecondOperand>>                                                     \
    _operator_name(const range<FirstAccessor> &first,                       \
                   const SecondOperand &second)                             \
    {                                                                       \
        return range<accessor::_operation_name<                             \
            ::gko::detail::operation_kind::range_by_scalar, FirstAccessor,  \
            SecondOperand>>(first.get_accessor(), second);                  \
    }                                                                       \
                                                                            \
    template <typename FirstOperand, typename SecondAccessor>               \
    GKO_ATTRIBUTES GKO_INLINE range<accessor::_operation_name<              \
        ::gko::detail::operation_kind::scalar_by_range, FirstOperand,       \
        SecondAccessor>>                                                    \
    _operator_name(const FirstOperand &first,                               \
                   const range<SecondAccessor> &second)                     \
    {                                                                       \
        return range<accessor::_operation_name<                             \
            ::gko::detail::operation_kind::scalar_by_range, FirstOperand,   \
            SecondAccessor>>(first, second.get_accessor());                 \
    }


#define GKO_DEFINE_SIMPLE_BINARY_OPERATION(_name, ...)                         \
    struct _name {                                                             \
    private:                                                                   \
        template <typename FirstOperand, typename SecondOperand>               \
        GKO_ATTRIBUTES static auto simple_evaluate_impl(                       \
            const FirstOperand &first, const SecondOperand &second)            \
            -> decltype(__VA_ARGS__)                                           \
        {                                                                      \
            return __VA_ARGS__;                                                \
        }                                                                      \
                                                                               \
    public:                                                                    \
        template <typename FirstAccessor, typename SecondAccessor,             \
                  typename... DimensionTypes>                                  \
        GKO_ATTRIBUTES static auto evaluate_range_by_range(                    \
            const FirstAccessor &first, const SecondAccessor &second,          \
            const DimensionTypes &... dims)                                    \
            -> decltype(simple_evaluate_impl(first(dims...), second(dims...))) \
        {                                                                      \
            return simple_evaluate_impl(first(dims...), second(dims...));      \
        }                                                                      \
                                                                               \
        template <typename FirstOperand, typename SecondAccessor,              \
                  typename... DimensionTypes>                                  \
        GKO_ATTRIBUTES static auto evaluate_scalar_by_range(                   \
            const FirstOperand &first, const SecondAccessor &second,           \
            const DimensionTypes &... dims)                                    \
            -> decltype(simple_evaluate_impl(first, second(dims...)))          \
        {                                                                      \
            return simple_evaluate_impl(first, second(dims...));               \
        }                                                                      \
                                                                               \
        template <typename FirstAccessor, typename SecondOperand,              \
                  typename... DimensionTypes>                                  \
        GKO_ATTRIBUTES static auto evaluate_range_by_scalar(                   \
            const FirstAccessor &first, const SecondOperand &second,           \
            const DimensionTypes &... dims)                                    \
            -> decltype(simple_evaluate_impl(first(dims...), second))          \
        {                                                                      \
            return simple_evaluate_impl(first(dims...), second);               \
        }                                                                      \
    };


namespace accessor {
namespace detail {


// binary arithmetic
GKO_DEFINE_SIMPLE_BINARY_OPERATION(add, first + second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(sub, first - second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(mul, first *second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(div, first / second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(mod, first % second);

// relational
GKO_DEFINE_SIMPLE_BINARY_OPERATION(less, first < second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(greater, first > second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(less_or_equal, first <= second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(greater_or_equal, first >= second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(equal, first == second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(not_equal, first != second);

// binary logical
GKO_DEFINE_SIMPLE_BINARY_OPERATION(logical_or, first || second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(logical_and, first &&second);

// binary bitwise
GKO_DEFINE_SIMPLE_BINARY_OPERATION(bitwise_or, first | second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(bitwise_and, first &second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(bitwise_xor, first ^ second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(left_shift, first << second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(right_shift, first >> second);

// common binary functions
GKO_DEFINE_SIMPLE_BINARY_OPERATION(max_operation, max(first, second));
GKO_DEFINE_SIMPLE_BINARY_OPERATION(min_operation, min(first, second));

// range binary functions
// TODO: change the way ranges are handled to allow an efficient, rank-K
// update-based implementation of matrix matrix multiply
struct mmul_operation {
    template <typename FirstAccessor, typename SecondAccessor,
              typename FirstDimension, typename SecondDimension,
              typename... Dimensions>
    GKO_ATTRIBUTES static auto evaluate_range_by_range(
        const FirstAccessor &first, const SecondAccessor &second,
        const FirstDimension &row, const SecondDimension &col,
        const Dimensions... rest)
        // use the type of a linear combination between
        // the elements of the first and the second operands
        -> decltype(first(row, 0, rest...) * second(0, col, rest...) +
                    first(row, 1, rest...) * second(1, col, rest...))
    {
        using result_type =
            decltype(first(row, 0, rest...) * second(0, col, rest...) +
                     first(row, 1, rest...) * second(1, col, rest...));
        GKO_ASSERT(first.length(1) == second.length(0));
        auto result = zero<result_type>();
        const auto size = first.length(1);
        for (auto i = zero(size); i < size; ++i) {
            result += first(row, i, rest...) * second(i, col, rest...);
        }
        return result;
    }
};


}  // namespace detail
}  // namespace accessor


// binary arithmetic
GKO_ENABLE_BINARY_RANGE_OPERATION(add, operator+, accessor::detail::add);
GKO_ENABLE_BINARY_RANGE_OPERATION(sub, operator-, accessor::detail::sub);
GKO_ENABLE_BINARY_RANGE_OPERATION(mul, operator*, accessor::detail::mul);
GKO_ENABLE_BINARY_RANGE_OPERATION(div, operator/, accessor::detail::div);
GKO_ENABLE_BINARY_RANGE_OPERATION(mod, operator%, accessor::detail::mod);

// relational
GKO_ENABLE_BINARY_RANGE_OPERATION(less, operator<, accessor::detail::less);
GKO_ENABLE_BINARY_RANGE_OPERATION(greater, operator>,
                                  accessor::detail::greater);
GKO_ENABLE_BINARY_RANGE_OPERATION(less_or_equal, operator<=,
                                  accessor::detail::less_or_equal);
GKO_ENABLE_BINARY_RANGE_OPERATION(greater_or_equal, operator>=,
                                  accessor::detail::greater_or_equal);
GKO_ENABLE_BINARY_RANGE_OPERATION(equal, operator==, accessor::detail::equal);
GKO_ENABLE_BINARY_RANGE_OPERATION(not_equal, operator!=,
                                  accessor::detail::not_equal);

// binary logical
GKO_ENABLE_BINARY_RANGE_OPERATION(logical_or, operator||,
                                  accessor::detail::logical_or);
GKO_ENABLE_BINARY_RANGE_OPERATION(logical_and, operator&&,
                                  accessor::detail::logical_and);

// binary bitwise
GKO_ENABLE_BINARY_RANGE_OPERATION(bitwise_or, operator|,
                                  accessor::detail::bitwise_or);
GKO_ENABLE_BINARY_RANGE_OPERATION(bitwise_and, operator&,
                                  accessor::detail::bitwise_and);
GKO_ENABLE_BINARY_RANGE_OPERATION(bitwise_xor, operator^,
                                  accessor::detail::bitwise_xor);
GKO_ENABLE_BINARY_RANGE_OPERATION(left_shift, operator<<,
                                  accessor::detail::left_shift);
GKO_ENABLE_BINARY_RANGE_OPERATION(right_shift, operator>>,
                                  accessor::detail::right_shift);

// common binary functions
GKO_ENABLE_BINARY_RANGE_OPERATION(max_operaton, max,
                                  accessor::detail::max_operation);
GKO_ENABLE_BINARY_RANGE_OPERATION(min_operaton, min,
                                  accessor::detail::min_operation);


// special binary range functions
GKO_ENABLE_BINARY_RANGE_OPERATION(mmul_operaton, mmul,
                                  accessor::detail::mmul_operation);


#undef GKO_DEFINE_SIMPLE_BINARY_OPERATION
#undef GKO_ENABLE_BINARY_RANGE_OPERATION


// implementations of accessors


namespace accessor {


template <typename ValueType, size_type Dimensionality>
class row_major {};  // TODO: implement accessor for other dimensionalities

template <typename ValueType>
class row_major<ValueType, 2> {
public:
    using value_type = ValueType;
    static constexpr size_type dimensionality = 2;
    using data_type = value_type *;

    GKO_ATTRIBUTES explicit row_major(data_type data, size_type num_rows,
                                      size_type num_cols, size_type stride)
        : data{data}, lengths{num_rows, num_cols}, stride{stride}
    {}

    GKO_ATTRIBUTES value_type &operator()(size_type row, size_type col) const
    {
        GKO_ASSERT(row < lengths[0]);
        GKO_ASSERT(col < lengths[1]);
        return data[row * stride + col];
    }

    GKO_ATTRIBUTES range<row_major> operator()(const span &rows,
                                               const span &cols) const
    {
        GKO_ASSERT(rows.begin <= rows.end);
        GKO_ASSERT(cols.begin <= cols.end);
        GKO_ASSERT(rows <= span::empty(lengths[0]));
        GKO_ASSERT(cols <= span::empty(lengths[1]));
        return range<row_major>(data + rows.begin * stride + cols.begin,
                                rows.end - rows.begin, cols.end - cols.begin,
                                stride);
    }

    GKO_ATTRIBUTES size_type length(size_type dimension) const
    {
        return dimension < 2 ? lengths[dimension] : 1;
    }

    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const
    {
        for (size_type i = 0; i < lengths[0]; ++i) {
            for (size_type j = 0; j < lengths[1]; ++j) {
                (*this)(i, j) = other(i, j);
            }
        }
    };

    const data_type data;
    const std::array<const size_type, dimensionality> lengths;
    const size_type stride;
};


}  // namespace accessor


}  // namespace gko


#endif  // GKO_CORE_BASE_RANGE_HPP_
