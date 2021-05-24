/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <iostream>
#include <map>
#include <string>


#include <omp.h>
#include <ginkgo/ginkgo.hpp>


// A CUDA kernel implementing the stencil, which will be used if running on the
// CUDA executor. Unfortunately, NVCC has serious problems interpreting some
// parts of Ginkgo's code, so the kernel has to be compiled separately.
template <typename ValueType, typename BoundaryType>
void stencil_kernel(std::size_t size, const BoundaryType *bd,
                    const ValueType *input, ValueType *output, std::size_t dimx,
                    std::size_t dimy, std::size_t dimz, bool init);

template <typename T>
inline T pitch(const T i, const T j, const T k, const T dim_x, const T dim_y,
               const T dim_z)
{
    return k * dim_y * dim_z + j * dim_y + i;
}

// A stencil matrix class representing the 3pt stencil linear operator.
// We include the gko::EnableLinOp mixin which implements the entire LinOp
// interface, except the two apply_impl methods, which get called inside the
// default implementation of apply (after argument verification) to perform the
// actual application of the linear operator. In addition, it includes the
// implementation of the entire PolymorphicObject interface.
//
// It also includes the gko::EnableCreateMethod mixin which provides a default
// implementation of the static create method. This method will forward all its
// arguments to the constructor to create the object, and return an
// std::unique_ptr to the created object.
static bool init = 0;
template <typename ValueType, typename BoundaryType>
class StencilMatrix
    : public gko::EnableLinOp<StencilMatrix<ValueType, BoundaryType>>,
      public gko::EnableCreateMethod<StencilMatrix<ValueType, BoundaryType>> {
public:
    // This constructor will be called by the create method. Here we initialize
    // the coefficients of the stencil.
    using vec = gko::matrix::Dense<ValueType>;
    using vec_bd = gko::matrix::Dense<BoundaryType>;

    StencilMatrix(std::shared_ptr<const gko::Executor> exec,
                  const vec_bd *bd = NULL, gko::size_type dimx = 0,
                  gko::size_type dimy = 0, gko::size_type dimz = 0)
        : gko::EnableLinOp<StencilMatrix>(exec,
                                          gko::dim<2>{dimx * dimy * dimz}),
          m_bd(bd),
          dimx(dimx),
          dimy(dimy),
          dimz(dimz)
    {}

protected:
    const vec_bd *m_bd;
    gko::size_type dimx;
    gko::size_type dimy;
    gko::size_type dimz;

    // Here we implement the application of the linear operator, x = A * b.
    // apply_impl will be called by the apply method, after the arguments have
    // been moved to the correct executor and the operators checked for
    // conforming sizes.
    //
    // For simplicity, we assume that there is always only one right hand side
    // and the stride of consecutive elements in the vectors is 1 (both of these
    // are always true in this example).
    void apply_impl(const gko::LinOp *input, gko::LinOp *output) const override
    {
        // we only implement the operator for dense RHS.
        // gko::as will throw an exception if its argument is not Dense.
        auto dense_input = gko::as<vec>(input);
        auto dense_output = gko::as<vec>(output);
        auto dense_bd = gko::as<vec_bd>(m_bd);

        // we need separate implementations depending on the executor, so we
        // create an operation which maps the call to the correct implementation
        struct stencil_operation : gko::Operation {
            stencil_operation(const vec_bd *bd, const vec *input, vec *output,
                              gko::size_type dimx = 0, gko::size_type dimy = 0,
                              gko::size_type dimz = 0)
                : m_m_bd(bd),
                  input(input),
                  output(output),
                  dimx(dimx),
                  dimy(dimy),
                  dimz(dimz)
            {}

            // OpenMP implementation
            void run(std::shared_ptr<const gko::OmpExecutor>) const override
            {
                auto input_values = input->get_const_values();
                auto output_values = output->get_values();
                auto bd_values = m_m_bd->get_const_values();

                // printf("\n ***** dim= %u, %u, %u", dimx, dimy, dimz);

                // assume h = 1
                const ValueType invh2 = ValueType(1.0);

                //#pragma omp parallel for
                for (gko::size_type k = 0; k < dimz; ++k) {
                    for (gko::size_type j = 0; j < dimy; ++j) {
                        for (gko::size_type i = 0; i < dimx; ++i) {
                            // printf(
                            //    "\n i= %u, j= %u, k= %u output= %f, input= %f,
                            //    " "bd= %f", i, j, k, output_values[pitch(i, j,
                            //    k, dimx, dimy, dimz)], input_values[pitch(i,
                            //    j, k, dimx, dimy, dimz)], bd_values[pitch(i,
                            //    j, k, dimx, dimy, dimz)]);

                            auto center_pitch =
                                pitch(i, j, k, dimx, dimy, dimz);
                            const ValueType center = input_values[center_pitch];
                            if (bd_values[center_pitch] == 0) {
                                if (!init) {
                                    output_values[center_pitch] = 0;
                                } else {
                                    output_values[center_pitch] = center;
                                }


                            } else {
                                ValueType sum = 0.0;
                                int numNeighb = 0;

                                if (i > 0) {
                                    ++numNeighb;
                                    sum += input_values[pitch(i - 1, j, k, dimx,
                                                              dimy, dimz)];
                                }

                                if (j > 0) {
                                    ++numNeighb;
                                    sum += input_values[pitch(i, j - 1, k, dimx,
                                                              dimy, dimz)];
                                }

                                if (k > 0) {
                                    ++numNeighb;
                                    sum += input_values[pitch(i, j, k - 1, dimx,
                                                              dimy, dimz)];
                                }

                                if (i < dimx - 1) {
                                    ++numNeighb;
                                    sum += input_values[pitch(i + 1, j, k, dimx,
                                                              dimy, dimz)];
                                }

                                if (j < dimy - 1) {
                                    ++numNeighb;
                                    sum += input_values[pitch(i, j + 1, k, dimx,
                                                              dimy, dimz)];
                                }

                                if (k < dimz - 1) {
                                    ++numNeighb;
                                    sum += input_values[pitch(i, j, k + 1, dimx,
                                                              dimy, dimz)];
                                }

                                output_values[center_pitch] =
                                    (-sum + static_cast<ValueType>(numNeighb) *
                                                center) *
                                    invh2;
                            }
                        }
                    }
                }

                /*printf("\n **** ");
                for (gko::size_type k = 0; k < dimz; ++k) {
                    for (gko::size_type j = 0; j < dimy; ++j) {
                        for (gko::size_type i = 0; i < dimx; ++i) {
                            printf("\n i= %u, j= %u, k= %u output= %f", i, j, k,
                                   output_values[pitch(i, j, k, dimx, dimy,
                                                       dimz)]);
                        }
                    }
                }*/
            }

            // CUDA implementation
            void run(std::shared_ptr<const gko::CudaExecutor>) const override
            {
                stencil_kernel<ValueType, BoundaryType>(
                    output->get_size()[0], m_m_bd->get_const_values(),
                    input->get_const_values(), output->get_values(), dimx, dimy,
                    dimz, init);
            }

            const vec_bd *m_m_bd;
            const vec *input;
            vec *output;

            gko::size_type dimx;
            gko::size_type dimy;
            gko::size_type dimz;
        };
        this->get_executor()->run(stencil_operation(
            dense_bd, dense_input, dense_output, dimx, dimy, dimz));
    }

    // There is also a version of the apply function which does the operation
    // output = alpha * A * input + beta * output. This function is commonly
    // used and can often be better optimized than implementing it using
    // output = A * input. However, for simplicity, we will implement it exactly
    // like that in this example.
    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *input,
                    const gko::LinOp *beta, gko::LinOp *output) const override
    {
        auto dense_input = gko::as<vec>(input);
        auto dense_output = gko::as<vec>(output);
        auto tmp_output = dense_output->clone();
        this->apply_impl(input, lend(tmp_output));
        dense_output->scale(beta);
        dense_output->add_scaled(alpha, lend(tmp_output));
        init = true;
    }

private:
};

template <typename ValueType>
void print_solution(const gko::matrix::Dense<ValueType> *u, uint32_t dimx,
                    uint32_t dimy, uint32_t dimz)
{
    for (uint32_t k = 0; k < dimz; ++k) {
        for (uint32_t j = 0; j < dimy; ++j) {
            for (uint32_t i = 0; i < dimx; ++i) {
                printf("\n i= %u, j= %u, k= %u u= %f", i, j, k,
                       u->get_const_values()[pitch(i, j, k, dimx, dimy, dimz)]);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    // Some shortcuts
    using ValueType = double;
    using BoundaryType =
        float;  // TODO this should be int8_t but this gives compile errors
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using bdvec = gko::matrix::Dense<BoundaryType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;

    gko::size_type max_num_iter = 1000;
    const RealValueType reduction_factor{1e-10};
    uint32_t dimx = 4;
    uint32_t dimy = 4;
    uint32_t dimz = 4;
    gko::size_type discretization_points = dimx * dimy * dimz;

    // executor where Ginkgo will perform the computation
    const auto exec =
        gko::CudaExecutor::create(0, gko::OmpExecutor::create(), true);
    // const auto exec = gko::OmpExecutor::create();

    // executor used by the application
    const auto app_exec = exec->get_master();

    // initialize vectors
    auto rhs = vec::create(app_exec, gko::dim<2>(discretization_points, 1));
    for (uint32_t k = 0; k < dimz; ++k) {
        for (uint32_t j = 0; j < dimy; ++j) {
            for (uint32_t i = 0; i < dimx; ++i) {
                rhs->get_values()[pitch(i, j, k, dimx, dimy, dimz)] = 0.;
            }
        }
    }

    auto u = vec::create(app_exec, gko::dim<2>(discretization_points, 1));
    ValueType bdZmin = -20.0;
    ValueType bdZMax = 20.0;
    for (uint32_t k = 0; k < dimz; ++k) {
        for (uint32_t j = 0; j < dimy; ++j) {
            for (uint32_t i = 0; i < dimx; ++i) {
                if (k == 0) {
                    u->get_values()[pitch(i, j, k, dimx, dimy, dimz)] = bdZmin;
                } else if (k == dimz - 1) {
                    u->get_values()[pitch(i, j, k, dimx, dimy, dimz)] = bdZMax;
                } else {
                    u->get_values()[pitch(i, j, k, dimx, dimy, dimz)] = 0.0;
                }
            }
        }
    }

    auto bd = bdvec::create(exec, gko::dim<2>(discretization_points, 1));
    for (uint32_t k = 0; k < dimz; ++k) {
        for (uint32_t j = 0; j < dimy; ++j) {
            for (uint32_t i = 0; i < dimx; ++i) {
                if (k == 0 || k == dimz - 1) {
                    bd->get_values()[pitch(i, j, k, dimx, dimy, dimz)] = 0;
                } else {
                    bd->get_values()[pitch(i, j, k, dimx, dimy, dimz)] = 1;
                }
            }
        }
    }


    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create(exec);

    auto iter_stop =
        gko::stop::Iteration::build().with_max_iters(max_num_iter).on(exec);
    auto tol_stop = gko::stop::ResidualNorm<ValueType>::build()
                        .with_reduction_factor(reduction_factor)
                        .on(exec);
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);


    // Generate solver and solve the system
    auto solver =
        cg::build()
            .with_criteria(gko::share(iter_stop), gko::share(tol_stop))
            .on(exec)
            ->generate(StencilMatrix<ValueType, BoundaryType>::create(
                exec, lend(bd), dimx, dimy, dimz));
    exec->synchronize();

    std::chrono::nanoseconds time(0);
    auto tic = std::chrono::steady_clock::now();
    solver->apply(lend(rhs), lend(u));
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);

    std::cout << "CG iteration count:     " << logger->get_num_iterations()
              << std::endl;
    std::cout << "CG execution time [ms]: "
              << static_cast<double>(time.count()) / 1000000.0 << std::endl;

    std::cout << "\nSolve complete.\n";

    // print_solution(lend(u), dimx, dimy, dimz);
}
