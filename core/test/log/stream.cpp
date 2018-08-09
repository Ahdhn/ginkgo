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

#include <core/log/stream.hpp>


#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>


#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>


namespace {


constexpr int num_iters = 10;
const std::string apply_str = "DummyLoggedClass::apply";


TEST(Stream, CatchesApply)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger =
        gko::log::Stream<>::create(exec, gko::log::Logger::apply_mask, out);

    logger->on<gko::log::Logger::apply>(apply_str);

    auto os = out.str();
    ASSERT_TRUE(os.find("starting apply function: " + apply_str) !=
                std::string::npos);
}


TEST(Stream, CatchesIterations)
{
    auto exec = gko::ReferenceExecutor::create();
    std::stringstream out;
    auto logger = gko::log::Stream<>::create(
        exec, gko::log::Logger::iteration_complete_mask, out);

    logger->on<gko::log::Logger::iteration_complete>(nullptr, num_iters,
                                                     nullptr, nullptr, nullptr);

    auto os = out.str();
    ASSERT_TRUE(os.find("iteration " + num_iters) != std::string::npos);
}


TEST(Stream, CatchesConvergence)
{
    std::stringstream out;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx =
        gko::initialize<gko::matrix::Dense<>>(4, {{1.0, 2.0, 3.0}}, exec);
    auto logger =
        gko::log::Stream<>::create(exec, gko::log::Logger::converged_mask, out);
    out << std::scientific << std::setprecision(4);

    logger->on<gko::log::Logger::converged>(num_iters, mtx.get());

    auto os = out.str();
    ASSERT_TRUE(os.find("converged at iteration " + num_iters) !=
                std::string::npos);
    ASSERT_TRUE(os.find("1.0") != std::string::npos);
    ASSERT_TRUE(os.find("2.0") != std::string::npos);
    ASSERT_TRUE(os.find("3.0") != std::string::npos);
}


}  // namespace
