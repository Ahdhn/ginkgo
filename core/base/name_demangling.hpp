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

#ifndef GKO_CORE_NAME_DEMANGLING_HPP
#define GKO_CORE_NAME_DEMANGLING_HPP


#include "config.hpp"

#ifdef HAVE_CXXABI_H
#include <cxxabi.h>
#endif  // HAVE_CXXABI_H


#include <string>


namespace gko {
namespace name_demangling {


/**
 * This is a function which uses `std::type_info` and demangling functionalities
 * when available to return the proper location at which this function is
 * called.
 *
 * @param type  the `std::type_info` of the object calling this function
 * @param func_name  the name of the function calling this (usually __func___)
 * @return properly formatted string representing the location of the call
 *
 * @see C++11 documentation [type.info] and [expr.typeid]
 */
inline std::string get_full_function_name(const std::type_info &type,
                                          const char *func_name)

{
#ifdef HAVE_CXXABI_H
    int status{};
    std::unique_ptr<char[], void (*)(void *)> demangled(
        abi::__cxa_demangle(type.name(), nullptr, nullptr, &status),
        &std::free);
    if (!status)
        return std::string() + demangled.get() + "::" + func_name;
    else
#endif
        return std::string() + type.name() + "::" + func_name;
}


}  // namespace name_demangling
}  // namespace gko


#endif  //  GKO_CORE_NAME_DEMANGLING_HPP
