#!/usr/bin/bash

# Extract script file name
SCRIPT_PATH=$(dirname $(realpath -s $0))
mkdir -p ${SCRIPT_PATH}/benchmarks/ginkgo

# Cardinality: 1
for r in 128 256 512 768; do
   echo ../../build/examples/custom-matrix-format/custom-matrix-format --cardinality 1 --domain_size ${r} --keeper_filename ${SCRIPT_PATH}/benchmarks/ginkgo/ginkgo_1d_${r}_1gpus_${t} --times 10
   ../../build/examples/custom-matrix-format/custom-matrix-format --cardinality 1 --domain_size ${r} --keeper_filename ${SCRIPT_PATH}/benchmarks/ginkgo/ginkgo_1d_${r}_1gpus_${t} --times 10
done

# Cardinality: 3
#for r in 282 355 448 564; do
#    echo ../../build/examples/custom-matrix-format/custom-matrix-format --cardinality 3 --domain_size ${r} --keeper_filename ${SCRIPT_PATH}/benchmarks/ginkgo/ginkgo_3d_${r}_1gpus_${t} --times 10   
#   ../build/Release/bin/solverPt_Poisson --cardinality 3 --domain_size ${r} --grid eGrid --keeper_filename ${SCRIPT_PATH}/benchmarks/Poisson/eGrid_OCC_3d_${r}_${n}gpus_${t} --gpus ${GPUS[@]:0:${n}} --times 1
#done
  


