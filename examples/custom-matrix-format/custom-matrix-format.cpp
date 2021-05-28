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
#include "perfkeeper.h"
#include "clipp.h"

int         DOMAIN_SIZE = 8;  
size_t      MAX_ITER = 100;   
double      TOL = 1e-10;        
double      ZMIN = -20.0;
double      ZMAX = +20.0;
int         CARDINALITY = 1;
std::string KEEPER_FILENAME = "keeper";
int         TIMES = 1;

template <typename T>
inline T pitch(const T i, const T j, const T k, const T c, const T dim_x, const T dim_y, const T dim_z)
{
    return c * dim_x * dim_y * dim_z + k * dim_y * dim_z + j * dim_y + i;
}

template<typename T>
void storeVTI(gko::matrix::Dense<T>* mat, std::size_t dimx, std::size_t dimy, std::size_t dimz, std::string filename) {

	std::string wholeExtent = std::string("0 ") + std::to_string(dimx) + std::string(" ") + std::string("0 ") + std::to_string(dimy) + std::string(" ") + std::string("0 ") + std::to_string(dimz) + std::string(" ");
	std::string spacing = std::to_string(1) + std::string(" ") + std::to_string(1) + std::string(" ") + std::to_string(1) + std::string(" ");

	std::ofstream out(filename, std::ios::out | std::ios::binary);
	if (!out.is_open()) {
		std::cout << "Error(!): Unable to open the file " << filename << "\n";
	}
	out.precision(17);

	out << "<?xml version=\"1.0\"?>" << std::endl;
	out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
	out << std::string("<ImageData WholeExtent=\"") + wholeExtent + std::string("\" Origin=\"") + std::to_string(0) + std::string(" ") + std::to_string(0) + std::string(" ") + std::to_string(0) + std::string("\" Spacing=\"") + spacing + std::string("\">\n");
	std::string pieceExtent = std::string("0 ") + std::to_string(dimx) + std::string(" ") + std::string("0 ") + std::to_string(dimy) + std::string(" ") + std::string("0 ") + std::to_string(dimz) + std::string(" ");
	out << std::string("<Piece Extent=\"") + pieceExtent + std::string("\" >") << std::endl;


	out << std::string("<CellData>\n");
	out << "<DataArray type=\"Float64\" NumberOfComponents=\"";
	out << 1;
	out << "\" Name=\"";
	out << "density";
	out << "\" format=\"ascii\">\n";
    for (std::size_t k = 0; k < dimz; k++) {
        for (std::size_t j = 0; j < dimy; j++) {
	        for (std::size_t i = 0; i < dimx; i++) {

				out << mat->at(pitch(i,j,k, std::size_t(0), dimx, dimy, dimz),0) << " ";
				out << "\t";
			}
			out << "\n";
		}
		out << "\n";
	}
	out << std::string("</DataArray>\n");
	out << std::string("</CellData>\n");


	out << std::string("</Piece>\n");
	out << std::string(" </ImageData>\n");
	out << std::string(" </VTKFile>\n");
	out.close();
}

// A CUDA kernel implementing the stencil, which will be used if running on the
// CUDA executor. Unfortunately, NVCC has serious problems interpreting some
// parts of Ginkgo's code, so the kernel has to be compiled separately.
template <typename ValueType, typename BoundaryType>
void stencil_kernel(std::size_t size, BoundaryType *bd,
                    const ValueType *input, ValueType *output, std::size_t dimx,
                    std::size_t dimy, std::size_t dimz, bool init);



// A stencil matrix class representing the 7pt stencil linear operator.
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
                  vec_bd *bd = NULL, gko::size_type dimx = 0,
                  gko::size_type dimy = 0, gko::size_type dimz = 0)
        : gko::EnableLinOp<StencilMatrix>(exec,
                                          gko::dim<2>{dimx * dimy * dimz}),
          m_bd(bd),
          dimx(dimx),
          dimy(dimy),
          dimz(dimz)
    {}

protected:
    vec_bd *m_bd;
    gko::size_type dimx;
    gko::size_type dimy;
    gko::size_type dimz;

    // Here we implement the application of the linear operator, x = A * b.
    // apply_impl will be called by the apply method, after the arguments have
    // been moved to the correct executor and the operators checked for
    // conforming sizes.
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
            stencil_operation(vec_bd *bd, const vec *input, vec *output,
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
                // assume h = 1
                const ValueType invh2 = ValueType(1.0);

                //#pragma omp parallel for
                for (gko::size_type k = 0; k < dimz; ++k) {
                    for (gko::size_type j = 0; j < dimy; ++j) {
                        for (gko::size_type i = 0; i < dimx; ++i) {
             
                            auto center_pitch = pitch(i, j, k, gko::size_type(0), dimx, dimy, dimz);
                            if(!init){
                                if (k == 0 || k == dimz - 1) {             
                                    m_m_bd->at(center_pitch,0) = 0;
                                } else {                                    
                                    m_m_bd->at(center_pitch,0) = 1;
                                }
                            }

                            const ValueType center = input->at(center_pitch,0);                            
                            if (m_m_bd->at(center_pitch,0) == 0) {
                                if (!init) {                                    
                                    output->at(center_pitch,0) = 0;
                                } else {                                    
                                    output->at(center_pitch,0) = center;
                                }

                            } else {
                                ValueType sum = 0.0;
                                int numNeighb = 0;

                                if (i > 0) {
                                    ++numNeighb;                                    
                                    sum += input->at(pitch(i - 1, j, k, gko::size_type(0),dimx,dimy, dimz),0);
                                }

                                if (j > 0) {
                                    ++numNeighb;                                    
                                    sum += input->at(pitch(i, j - 1, k, gko::size_type(0),dimx,dimy, dimz),0);
                                }

                                if (k > 0) {
                                    ++numNeighb;                                    
                                    sum += input->at(pitch(i, j, k - 1, gko::size_type(0),dimx,dimy, dimz),0);
                                }

                                if (i < dimx - 1) {
                                    ++numNeighb;                                    
                                    sum += input->at(pitch(i + 1, j, k, gko::size_type(0),dimx,dimy, dimz),0);
                                }

                                if (j < dimy - 1) {
                                    ++numNeighb;                                    
                                    sum += input->at(pitch(i, j + 1, k, gko::size_type(0),dimx, dimy, dimz),0);
                                }

                                if (k < dimz - 1) {
                                    ++numNeighb;                                    
                                    sum += input->at(pitch(i, j, k + 1, gko::size_type(0),dimx,dimy, dimz),0);
                                }

                                output->at(center_pitch,0) =
                                    (-sum + static_cast<ValueType>(numNeighb) * center) * invh2;
                            }
                        }
                    }
                }
            }

            // CUDA implementation
            void run(std::shared_ptr<const gko::CudaExecutor>) const override
            {
                stencil_kernel<ValueType, BoundaryType>(
                    output->get_size()[0], m_m_bd->get_values(),
                    input->get_const_values(), output->get_values(), dimx, dimy,
                    dimz, init);
            }

            vec_bd *m_m_bd;
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
                       u->get_const_values()[pitch(i, j, k, 0u,dimx, dimy, dimz)]);
            }
        }
    }
}

inline void run()
{
    // Some shortcuts
    using ValueType = double;
    using BoundaryType = float;  // TODO this should be int8_t but this gives compile errors
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using bdvec = gko::matrix::Dense<BoundaryType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    
    Neon::PerfKeeper_t keeper;

    auto recordId = keeper.addRecord("Poisson_Ginkgo_" + std::to_string(CARDINALITY) + "D_1GPUs", Neon::PerfKeeper_t::addDefaultInfo);
    keeper.addStaticAtt(recordId, "maxIters", std::to_string(MAX_ITER), true);
    keeper.addStaticAtt(recordId, "voxelDomain", std::to_string(DOMAIN_SIZE)+ "_" + std::to_string(DOMAIN_SIZE) + "_" + std::to_string(DOMAIN_SIZE), true);
    keeper.addStaticAtt(recordId, "cardinality", std::to_string(CARDINALITY), true);
    keeper.addStaticAtt(recordId, "numGPUs", std::to_string(1), true, "nGPUs");
    keeper.addStaticAtt(recordId, "absTol", std::to_string(TOL), true);

    auto dynAttTimeSol = keeper.addDynamicAtt(recordId, "TimeToSolution", "ms");
    auto dynAttTimeTotal = keeper.addDynamicAtt(recordId, "TimeTotal", "ms");
    //auto dynAttResidualStart = keeper.addDynamicAtt(recordId, "ResidualStart", "");
    //auto dynAttResidualFinal = keeper.addDynamicAtt(recordId, "ResidualFinal", "");
    auto dynAttIterationsTaken = keeper.addDynamicAtt(recordId, "IterationsTaken", "");

    const RealValueType reduction_factor{TOL};
    uint32_t dimx = DOMAIN_SIZE;
    uint32_t dimy = DOMAIN_SIZE;
    uint32_t dimz = DOMAIN_SIZE;
    uint32_t cardinality = CARDINALITY;
    gko::size_type discretization_points = dimx * dimy * dimz * cardinality;

    // executor where Ginkgo will perform the computation
    const auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create(), true);
    //const auto exec = gko::OmpExecutor::create();

    // executor used by the application
    const auto app_exec = exec->get_master();

    for (int t = 0; t < TIMES; ++t) {
        // initialize vectors        
        auto rhs = vec::create(app_exec, gko::dim<2>(discretization_points, 1));    
        for (uint32_t k = 0; k < dimz; ++k) {
            for (uint32_t j = 0; j < dimy; ++j) {
                for (uint32_t i = 0; i < dimx; ++i) {
                    rhs->at(pitch(i, j, k, 0u, dimx, dimy, dimz),0) = 0.;                
                }
            }
        }

        auto u = vec::create(app_exec, gko::dim<2>(discretization_points, 1));    
        ValueType bdZmin(ZMIN);
        ValueType bdZMax(ZMAX);
        for (uint32_t k = 0; k < dimz; ++k) {
            for (uint32_t j = 0; j < dimy; ++j) {
                for (uint32_t i = 0; i < dimx; ++i) {
                    if (k == 0) {
                        u->at(pitch(i, j, k, 0u, dimx, dimy, dimz),0) = bdZmin;                    
                    } else if (k == dimz - 1) {
                        u->at(pitch(i, j, k, 0u, dimx, dimy, dimz),0) = bdZMax;                    
                    } else {
                        u->at(pitch(i, j, k, 0u, dimx, dimy, dimz),0) = 0.0;                    
                    }
                }
            }
        }

        auto bd = bdvec::create(exec, gko::dim<2>(discretization_points, 1));

        std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
            gko::log::Convergence<ValueType>::create(exec);

        auto iter_stop =
            gko::stop::Iteration::build().with_max_iters(MAX_ITER).on(exec);
        auto tol_stop = gko::stop::ResidualNorm<ValueType>::build()
                            .with_reduction_factor(reduction_factor)
                            .on(exec);
        iter_stop->add_logger(logger);
        tol_stop->add_logger(logger);


        // Generate solver and solve the system
        auto solver =
            cg::build()
                .with_criteria(gko::share(iter_stop), gko::share(tol_stop)).on(exec)
                ->generate(StencilMatrix<ValueType, BoundaryType>::create(exec, lend(bd), dimx, dimy, dimz));
        exec->synchronize();


        std::chrono::nanoseconds time(0);
        auto tic = std::chrono::steady_clock::now();    
        solver->apply(lend(rhs), lend(u));    
        auto toc = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);


        auto time_ms = static_cast<double>(time.count()) / 1000000.0 ;
        std::cout << "CG iteration count: " << logger->get_num_iterations() << std::endl;
        std::cout << "CG execution time [ms]: " << time_ms << std::endl;
        //std::cout << "CG converged: " << (logger->has_converged()? "yes":"no") << std::endl;
        std::cout << "Solve complete.\n";
        
        keeper.updateDynamicAtt(recordId, dynAttTimeSol, time_ms);
        keeper.updateDynamicAtt(recordId, dynAttTimeTotal, time_ms);
        //keeper.updateDynamicAtt(recordId, dynAttResidualStart, result.residualStart);
        //keeper.updateDynamicAtt(recordId, dynAttResidualFinal, result.residualEnd);
        keeper.updateDynamicAtt(recordId, dynAttIterationsTaken, double(logger->get_num_iterations()));

    }
    keeper.saveJson(KEEPER_FILENAME);

    //print_solution(lend(u), dimx, dimy, dimz);
    //storeVTI(lend(u), dimx, dimy, dimz, "solution.vti");    
}


int main(int argc, char *argv[]){

    // CLI for performance test
    auto cli =
        (clipp::option("--cardinality") & clipp::value("cardinality", CARDINALITY) % "Must be 1 or 3",
         clipp::option("--domain_size") & clipp::integer("domain_size", DOMAIN_SIZE) % "Voxels along each dimension of the cube domain",
         clipp::option("--max_iter") & clipp::integer("max_iter", MAX_ITER) % "Maximum solver iterations",
         clipp::option("--tol") & clipp::number("tol", TOL) % "Absolute tolerance for convergence",
         clipp::option("--keeper_filename ") & clipp::value("keeper_filename", KEEPER_FILENAME) % "Output perf keeper filename",
         clipp::option("--times ") & clipp::integer("times", TIMES) % "Times to run the experiment");

    if (!clipp::parse(argc, argv, cli)) {
        auto fmt = clipp::doc_formatting{}.doc_column(31);
        std::cout << make_man_page(cli, argv[0], fmt) << '\n';
        return -1;
    }
    run();
}