#ifndef MSPARSE_CORE_EXCEPTION_HELPERS_HPP_
#define MSPARSE_CORE_EXCEPTION_HELPERS_HPP_


#include "exception.hpp"


#include <typeinfo>


namespace msparse {


/**
 * Mark the function as not implemented.
 *
 * Attempts to call this function will result in a runtime error of type
 * NotImplemented.
 */
#define NOT_IMPLEMENTED \
{ throw ::msparse::NotImplemented(__FILE__, __LINE__, __func__); }


/**
 * Create a NotSupported exception.
 * This macro sets the correct information about the location of the error
 * and fills the exception with data about _obj.
 *
 * @param _obj  the object referenced by NotSupported exception
 *
 * @return NotSupported
 */
#define NOT_SUPPORTED(_obj) \
    ::msparse::NotSupported(__FILE__, __LINE__, __func__, typeid(_obj).name())


/**
 * Assert that _operator can be applied to _vectors.
 *
 * @throw DimensionMismatch  if _operator cannot be applied to _vectors.
 */
#define ASSERT_CONFORMANT(_operator, _vectors) \
    if ((_operator)->get_num_cols() != (_vectors)->get_num_rows()) { \
        throw ::msparse::DimensionMismatch(__FILE__, __LINE__, __func__, \
                (_operator)->get_num_rows(), (_operator)->get_num_cols(), \
                (_vectors)->get_num_rows(), (_vectors)->get_num_cols()); \
    }


namespace detail {


template <typename T>
inline T ensure_allocated_impl(
        T ptr, const std::string &file, int line, const std::string &dev,
        size_type size)
{
    if (ptr == nullptr) {
        throw AllocationError(file, line, dev, size);
    }
    return ptr;
}


}  // namespace detail


/**
 * Ensure the memory referenced by _ptr is allocated.
 * 
 * @param _ptr  the result of the allocation, if it is a nullptr, an exception
 *              is thrown
 * @param _dev  the device where the data was allocated, used to provide
 *              additional information in the error message
 * @param _size  the size in bytes of the allocated memory, used to provide
 *               additional information in the error message
 *
 * @throw AllocationError  if the pointer equals nullptr
 *
 * @return _ptr
 */
#define ENSURE_ALLOCATED(_ptr, _dev, _size) \
    ::msparse::detail::ensure_allocated_impl( \
            _ptr, __FILE__, __LINE__, _dev, _size)


}  // namespace msparse


#endif // MSPARSE_CORE_EXCEPTION_HELPERS_HPP_

