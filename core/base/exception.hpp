#ifndef GKO_CORE_EXCEPTION_HPP_
#define GKO_CORE_EXCEPTION_HPP_


#include "core/base/types.hpp"


#include <exception>
#include <string>


namespace gko {


/**
 * The Error class is used to report exceptional behaviour in library
 * functions. Ginkgo uses C++ exception mechanism to this end, and the
 * Error class represents a base class for all types of errors. The exact list
 * of errors which could occur during the execution of a certain library
 * routine is provided in the documentation of that routine, along with a short
 * description of the situation when that error can occur.
 * During runtime, these errors can be detected by using standard C++ try-catch
 * blocks, and a human-readable error description can be obtained by calling
 * the Error::what() method.
 *
 * As an example, trying to compute a matrix-vector product with arguments of
 * incompatible size will result in a DimensionMismatch error, which is
 * demonstrated in the following program.
 *
 * ```cpp
 * #include <ginkgo.h>
 * #include <iostream>
 *
 * using namespace gko;
 *
 * int main()
 * {
 *     auto cpu = create<CpuExecutor>();
 *     auto A = randn_fill<matrix::Csr<float>>(5, 5, 0f, 1f, cpu);
 *     auto x = fill<matrix::Dense<float>>(6, 1, 1f, cpu);
 *     try {
 *         auto y = apply(A.get(), x.get());
 *     } catch(Error e) {
 *         // an error occured, write the message to screen and exit
 *         std::cout << e.what() << std::endl;
 *         return -1;
 *     }
 *     return 0;
 * }
 * ```
 */
class Error : public std::exception {
public:
    Error(const std::string &file, int line, const std::string &what)
        : what_(file + ":" + std::to_string(line) + ": " + what)
    {}

    /**
     * Returns a human-readable string with a more detailed description of the
     * error.
     */
    virtual const char *what() const noexcept override { return what_.c_str(); }

private:
    const std::string what_;
};


/**
 * NotImplemented is thrown in case an operation has not yet
 * been implemented (but will be implemented in the future).
 */
class NotImplemented : public Error {
public:
    NotImplemented(const std::string &file, int line, const std::string &func)
        : Error(file, line, func + " is not implemented")
    {}
};


/**
 * NotCompiled is thrown when attempting to call an operation which is a part of
 * a module that was not compiled on the system.
 */
class NotCompiled : public Error {
public:
    NotCompiled(const std::string &file, int line, const std::string &func,
                const std::string &module)
        : Error(file, line,
                "feature " + func + " is part of the " + module +
                    " module, which is not compiled on this system")
    {}
};


/**
 * NotSupported is thrown in case it is not possible to
 * perform the requested operation on the given object type.
 */
class NotSupported : public Error {
public:
    NotSupported(const std::string &file, int line, const std::string &func,
                 const std::string &obj_type)
        : Error(file, line,
                "Operation " + func + " does not support parameters of type " +
                    obj_type)
    {}
};


/**
 * DimensionMismatch is thrown if a LinOp operation is being applied on
 * operators of incompatible size.
 */
class DimensionMismatch : public Error {
public:
    DimensionMismatch(const std::string &file, int line,
                      const std::string &func, const std::string &first_name,
                      size_type first_rows, size_type first_cols,
                      const std::string &second_name, size_type second_rows,
                      size_type second_cols, const std::string &clarification)
        : Error(file, line,
                func + ": attempting to combine operators " + first_name +
                    " [" + std::to_string(first_rows) + " x " +
                    std::to_string(first_cols) + "] and " + second_name + " [" +
                    std::to_string(second_rows) + " x " +
                    std::to_string(second_cols) + "]: " + clarification)
    {}
};


/**
 * NotFound is thrown if a requested Attachement is not found in the
 * attachement list of the LinOp.
 */
class NotFound : public Error {
public:
    NotFound(const std::string &file, int line, const std::string &func,
             const std::string &what)
        : Error(file, line, func + ": " + what)
    {}
};


/**
 * AllocationError is thrown if a memory allocation fails.
 */
class AllocationError : public Error {
public:
    AllocationError(const std::string &file, int line,
                    const std::string &device, size_type bytes)
        : Error(file, line,
                device + ": failed to allocate memory block of " +
                    std::to_string(bytes) + "B")
    {}
};


}  // namespace gko


#endif  // GKO_CORE_EXCEPTION_HPP_
