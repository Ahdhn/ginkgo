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

#include <core/solver/xxsolverxx.hpp>


#include <gtest/gtest.h>


#include <random>


#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>
#include <core/solver/xxsolverxx_kernels.hpp>
#include <core/test/utils.hpp>

namespace {


// This is example code for the CG case - has to be modified for the new solver
/*


class Xxsolverxx : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    Xxsolverxx() : rand_engine(30) {}

    void SetUp()
    {
        ASSERT_GT(gko::GpuExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        gpu = gko::GpuExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (gpu != nullptr) {
            ASSERT_NO_THROW(gpu->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            ref, num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine);
    }

    void initialize_data()
    {
        int m = 97;
        int n = 43;
        b = gen_mtx(m, n);
        r = gen_mtx(m, n);
        z = gen_mtx(m, n);
        p = gen_mtx(m, n);
        q = gen_mtx(m, n);
        x = gen_mtx(m, n);
        beta = gen_mtx(1, n);
        prev_rho = gen_mtx(1, n);
        rho = gen_mtx(1, n);

        d_b = Mtx::create(gpu);
        d_b->copy_from(b.get());
        d_r = Mtx::create(gpu);
        d_r->copy_from(r.get());
        d_z = Mtx::create(gpu);
        d_z->copy_from(z.get());
        d_p = Mtx::create(gpu);
        d_p->copy_from(p.get());
        d_q = Mtx::create(gpu);
        d_q->copy_from(q.get());
        d_x = Mtx::create(gpu);
        d_x->copy_from(x.get());
        d_beta = Mtx::create(gpu);
        d_beta->copy_from(beta.get());
        d_prev_rho = Mtx::create(gpu);
        d_prev_rho->copy_from(prev_rho.get());
        d_rho = Mtx::create(gpu);
        d_rho->copy_from(rho.get());

        r_result = Mtx::create(ref);
        z_result = Mtx::create(ref);
        p_result = Mtx::create(ref);
        q_result = Mtx::create(ref);
        x_result = Mtx::create(ref);
        beta_result = Mtx::create(ref);
        prev_rho_result = Mtx::create(ref);
        rho_result = Mtx::create(ref);
    }

    void copy_back_data()
    {
        r_result->copy_from(d_r.get());
        z_result->copy_from(d_z.get());
        p_result->copy_from(d_p.get());
        q_result->copy_from(d_q.get());
        x_result->copy_from(d_x.get());
        beta_result->copy_from(d_beta.get());
        prev_rho_result->copy_from(d_prev_rho.get());
        rho_result->copy_from(d_rho.get());
    }

    void make_symetric(Mtx *mtx)
    {
        for (int i = 0; i < mtx->get_num_rows(); ++i) {
            for (int j = i + 1; j < mtx->get_num_cols(); ++j) {
                mtx->at(i, j) = mtx->at(j, i);
            }
        }
    }

    void make_diag_dominant(Mtx *mtx)
    {
        using std::abs;
        for (int i = 0; i < mtx->get_num_rows(); ++i) {
            auto sum = gko::zero<Mtx::value_type>();
            for (int j = 0; j < mtx->get_num_cols(); ++j) {
                sum += abs(mtx->at(i, j));
            }
            mtx->at(i, i) = sum;
        }
    }

    void make_spd(Mtx *mtx)
    {
        make_symetric(mtx);
        make_diag_dominant(mtx);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::GpuExecutor> gpu;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> r;
    std::unique_ptr<Mtx> z;
    std::unique_ptr<Mtx> p;
    std::unique_ptr<Mtx> q;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> prev_rho;
    std::unique_ptr<Mtx> rho;

    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_r;
    std::unique_ptr<Mtx> d_z;
    std::unique_ptr<Mtx> d_p;
    std::unique_ptr<Mtx> d_q;
    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_beta;
    std::unique_ptr<Mtx> d_prev_rho;
    std::unique_ptr<Mtx> d_rho;

    std::unique_ptr<Mtx> r_result;
    std::unique_ptr<Mtx> z_result;
    std::unique_ptr<Mtx> p_result;
    std::unique_ptr<Mtx> q_result;
    std::unique_ptr<Mtx> x_result;
    std::unique_ptr<Mtx> beta_result;
    std::unique_ptr<Mtx> prev_rho_result;
    std::unique_ptr<Mtx> rho_result;
};


TEST_F(Xxsolverxx, GpuXxsolverxxInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::xxsolverxx::initialize(b.get(), r.get(), z.get(),
p.get(), q.get(), prev_rho.get(), rho.get());
    gko::kernels::gpu::xxsolverxx::initialize(d_b.get(), d_r.get(), d_z.get(),
                                      d_p.get(), d_q.get(), d_prev_rho.get(),
                                      d_rho.get());

    copy_back_data();
    ASSERT_MTX_NEAR(r_result, r, 1e-14);
    ASSERT_MTX_NEAR(z_result, z, 1e-14);
    ASSERT_MTX_NEAR(p_result, p, 1e-14);
    ASSERT_MTX_NEAR(q_result, q, 1e-14);
    ASSERT_MTX_NEAR(prev_rho_result, prev_rho, 1e-14);
    ASSERT_MTX_NEAR(rho_result, rho, 1e-14);
}


TEST_F(Xxsolverxx, GpuXxsolverxxStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::xxsolverxx::step_1(p.get(), z.get(), rho.get(),
                                        prev_rho.get());
    gko::kernels::gpu::xxsolverxx::step_1(d_p.get(), d_z.get(), d_rho.get(),
                                  d_prev_rho.get());
    copy_back_data();

    ASSERT_MTX_NEAR(p_result, p, 1e-14);
    ASSERT_MTX_NEAR(z_result, z, 1e-14);
}


TEST_F(Xxsolverxx, GpuXxsolverxxStep2IsEquivalentToRef)
{
    initialize_data();
    gko::kernels::reference::xxsolverxx::step_2(x.get(), r.get(), p.get(),
q.get(), beta.get(), rho.get());
    gko::kernels::gpu::xxsolverxx::step_2(d_x.get(), d_r.get(), d_p.get(),
d_q.get(), d_beta.get(), d_rho.get());

    copy_back_data();

    ASSERT_MTX_NEAR(x_result, x, 1e-14);
    ASSERT_MTX_NEAR(r_result, r, 1e-14);
    ASSERT_MTX_NEAR(p_result, p, 1e-14);
    ASSERT_MTX_NEAR(q_result, q, 1e-14);
}


TEST_F(Xxsolverxx, ApplyIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50);
    make_spd(mtx.get());
    auto x = gen_mtx(50, 3);
    auto b = gen_mtx(50, 3);
    auto d_mtx = Mtx::create(gpu);
    d_mtx->copy_from(mtx.get());
    auto d_x = Mtx::create(gpu);
    d_x->copy_from(x.get());
    auto d_b = Mtx::create(gpu);
    d_b->copy_from(b.get());
    auto xxsolverxx_factory = gko::solver::XxsolverxxFactory<>::create(ref, 50,
1e-14); auto d_xxsolverxx_factory =
gko::solver::XxsolverxxFactory<>::create(gpu, 50, 1e-14); auto solver =
xxsolverxx_factory->generate(std::move(mtx)); auto d_solver =
d_xxsolverxx_factory->generate(std::move(d_mtx));

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    auto result = Mtx::create(ref);
    result->copy_from(d_x.get());

    ASSERT_MTX_NEAR(result, x, 1e-14);
}


*/


}  // namespace
