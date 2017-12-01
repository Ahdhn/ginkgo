#include <gpu/base/executor.cpp>


#include <type_traits>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>


#include <cuda_runtime.h>

#include <gpu/test/base/gpu_kernel.cu>
namespace {


using exec_ptr = std::shared_ptr<gko::Executor>;


TEST(GpuExecutor, AllocatesAndFreesMemory)
{
    const int num_elems = 10;
    exec_ptr cpu = gko::CpuExecutor::create();
    exec_ptr gpu = gko::GpuExecutor::create(0, cpu);
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = gpu->alloc<int>(num_elems));
    ASSERT_NO_THROW(gpu->free(ptr));
}


TEST(GpuExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    exec_ptr cpu = gko::CpuExecutor::create();
    exec_ptr gpu = gko::GpuExecutor::create(0, cpu);
    int *ptr = nullptr;

    ASSERT_THROW(ptr = gpu->alloc<int>(num_elems), gko::AllocationError);

    gpu->free(ptr);
}


TEST(GpuExecutor, CopiesDataFromCpu)
{
    double orig[] = {3, 8};

    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr cpu = gko::CpuExecutor::create();
    exec_ptr gpu = gko::GpuExecutor::create(0, cpu);
    double *d_copy = gpu->alloc<int>(num_elems);
    double *copy = cpu->alloc<int>(num_elems);

    gpu->copy_from(cpu, num_elems, orig, copy);

    run_on_gpu(num_elems, d_copy);
    cpu->copy_from(gpu, num_elems, d_copy, copy) EXPECT_EQ(2.5, copy[0]);
    EXPECT_EQ(5, copy[1]);

    gpu->free(d_copy);
    cpu->free(copy);
}


}  // namespace
