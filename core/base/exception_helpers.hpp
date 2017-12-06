#ifndef GKO_CORE_EXCEPTION_HELPERS_HPP_
#define GKO_CORE_EXCEPTION_HELPERS_HPP_


#include "core/base/exception.hpp"


#include <typeinfo>


namespace gko {


/**
 * Marks a function as not yet implemented.
 *
 * Attempts to call this function will result in a runtime error of type
 * NotImplemented.
 */
#define NOT_IMPLEMENTED                                            \
    {                                                              \
        throw ::gko::NotImplemented(__FILE__, __LINE__, __func__); \
    }


/**
 * Marks a function as not compiled.
 *
 * Attempts to call this function will result in a runtime error of type
 * NotCompiled
 *
 * @param _module  the module which should be compiled to enable the function
 */
#define NOT_COMPILED(_module)                                             \
    {                                                                     \
        throw ::gko::NotCompiled(__FILE__, __LINE__, __func__, #_module); \
    }


/**
 * Creates a NotSupported exception.
 * This macro sets the correct information about the location of the error
 * and fills the exception with data about _obj.
 *
 * @param _obj  the object referenced by NotSupported exception
 *
 * @return NotSupported
 */
#define NOT_SUPPORTED(_obj) \
    ::gko::NotSupported(__FILE__, __LINE__, __func__, typeid(_obj).name())


/**
 * Asserts that _operator can be applied to _vectors.
 *
 * @throw DimensionMismatch  if _operator cannot be applied to _vectors.
 */
#define ASSERT_CONFORMANT(_operator, _vectors)                         \
    if ((_operator)->get_num_cols() != (_vectors)->get_num_rows()) {   \
        throw ::gko::DimensionMismatch(                                \
            __FILE__, __LINE__, __func__, (_operator)->get_num_rows(), \
            (_operator)->get_num_cols(), (_vectors)->get_num_rows(),   \
            (_vectors)->get_num_cols());                               \
    }


/**
 * Throws CUDA Error based on the value of the error code returned.
 * Wrapper for the actual CUDA error calling routine. Required to
 * allow for piggyback compilation of CUDA code.
 *
 * @param errcode The error code returned from the cuda routine.
 */
#define CUDA_ERROR(_errcode) \
    ::gko::CudaError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Asserts that a cuda library call completed without errors.
 *
 * @param _cuda_call  a library call expression
 */
#define ASSERT_NO_CUDA_ERRORS(_cuda_call) \
    auto _errcode = _cuda_call;           \
    if (_errcode != cudaSuccess) {        \
        throw CUDA_ERROR(_errcode);       \
    }


namespace detail {


template <typename T>
inline T ensure_allocated_impl(T ptr, const std::string &file, int line,
                               const std::string &dev, size_type size)
{
    if (ptr == nullptr) {
        throw AllocationError(file, line, dev, size);
    }
    return ptr;
}


}  // namespace detail


/**
 * Ensures the memory referenced by _ptr is allocated.
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
    ::gko::detail::ensure_allocated_impl(_ptr, __FILE__, __LINE__, _dev, _size)


}  // namespace gko


#endif  // GKO_CORE_EXCEPTION_HELPERS_HPP_
