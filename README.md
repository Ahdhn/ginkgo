Ginkgo
=======

Ginkgo is a numerical linear algebra library targeting manycore architectures.


Prerequisites
-------------

### Linux

For Ginkgo core library:

*   _cmake 3.1+_
*   C++11 compliant compiler, one of:
    *   _gcc 5.4.0+_
    *   _clang 3.3+_ (__TODO__: verify, works with 5.0)

The Ginkgo GPU module has the following __additional__ requirements:

*   _cmake 3.8+_
*   _CUDA 7.0+_ (__TODO__: verify, works with 8.0)
*   Any host compiler restrictions your version of CUDA may impose also apply
    here. For the newest CUDA version, this information can be found in the
    [CUDA installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)

In addition, if you want to contribute code to Ginkgo, you will also need the
following:

*   _clang-format 5.0.1+_ (ships as part of _clang_)
*   __NOTE:__ If you want to use _clang_ as your compiler and develop Ginkgo,
    you'll currently need two versions _clang_: _clang 4.0.0_ or older, as this
    is this version supporetd by the CUDA 9.1 toolkit, and _clang 5.0.1_ or
    newer, which will not be used for compilation, but only provide the
    _clang-format_ utility


### Mac OS

For Ginkgo core library:

*   _cmake 3.1+_
*   C++11 compliant compiler, one of:
    *   _gcc 5.4.0+_ (__TODO__: verify)
    *   _clang 3.3+_ (__TODO__: verify)
    *   _Apple LLVM 8.0+_ (__TODO__: verify)

The Ginkgo GPU module has the following __additional__ requirements:

*   _cmake 3.8+_
*   _CUDA 7.0+_ (__TODO__: verify)
*   Any host compiler restrictions your version of CUDA may impose also apply
    here. For the newest CUDA version, this information can be found in the
    [CUDA installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html)
In addition, if you want to contribute code to Ginkgo, you will also need the
following:

*   _clang-format 5.0.1+_ (ships as part of _clang_)
*   __NOTE:__ If you want to use _clang_ as your compiler and develop Ginkgo,
    you'll currently need two versions _clang_: _clang 4.0.0_ or older, as this
    is this version supporetd by the CUDA 9.1 toolkit, and _clang 5.0.1_ or
    newer, which will not be used for compilation, but only provide the
    _clang-format_ utility


### Windows

Windows is currently not supported, but we are working on porting the library
there. If you are interested in helping us with this effort, feel free to
contact one of the developers. (The library itself doesn't use any non-standard
C++ features, so most of the effort here is in modifying the build system.)

__TODO:__ Some restrictions will also apply on the version of C and C++ standard
libraries installed on the system. We need to investigate this further.


Installation
------------

Use the standard cmake build procedure:

```sh
mkdir build; cd build
cmake -G "Unix Makefiles" [OPTIONS] .. && make
```

Replace `[OPTIONS]` with desired cmake options for your build.
Ginkgo adds the following additional switches to control what is being built:

*   `-DDEVEL_TOOLS={ON, OFF}` sets up the build system for development
    (requires clang-format, will also download git-cmake-format),
    default is `ON`
*   `-DBUILD_TESTS={ON, OFF}` builds Ginkgo's tests
    (will download googletest), default is `ON`
*   `-DBUILD_REFERENCE={ON, OFF}` build reference implementations of the
    kernels, usefull for testing, default os `OFF`
*   `-DBUILD_CPU={ON, OFF}` builds optimized cpu versions of the kernels,
    default is `OFF`
*   `-DBUILD_GPU={ON, OFF}` builds optimized gpu versions of the kernels
    (requires CUDA), default is `OFF`
*   `-DSET_CUDA_HOST_COMPILER={ON, OFF}` instructs the build system to
    explicitly set CUDA's host compiler to match the commpiler used to build the
    the rest of the library (otherwise the nvcc toolchain uses its default host
    compiler). Setting this option may help if you're experiencing linking
    errors due to ABI incompatibilities. The default is `OFF`.

For example, to build everything (in debug mode), use:

```cmake
mkdir build; cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug -DDEVEL_TOOLS=ON \
      -DBUILD_TESTS=ON -DBUILD_REFERENCE=ON -DBUILD_CPU=ON -DBUILD_GPU=ON  ..
make
```

__NOTE:__ Currently, the only verified CMake generator is `Unix Makefiles`.
Other generators may work, but are not officially supported.

### Running the unit tests

You need to compile ginkgo with `-DBUILD_TESTS=ON` option to be able to run the
tests. Use the following command inside the build folder to run all tests:

```sh
make test
```

The output should contain several lines of the form:

```
     Start  1: path/to/test
 1/13 Test  #1: path/to/test .............................   Passed    0.01 sec
```

To run only a specific test and see more details results (e.g. if a test failed)
run the following from the build folder:

```sh
./path/to/test
```

where `path/to/test` is the path returned by `make test`.


### Installing ginkgo

__TODO:__ Install step is not yet set up.
