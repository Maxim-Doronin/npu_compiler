# Code coverage

VPU plug-in supports `gcov` tool for code coverage. Usage:

1. Append `-D VPUX_CODE_COVERAGE=GCOV` option to cmake invocation.
2. Build the project and run any application. For example:
```
$ ./vpux-opt --split-input-file --init-compiler="vpu-arch=NPU37XX" --fold-relu-before-fq fold_relu_before_fq.mlir
```
3. Locate '*.gcda' data file.
```
$ find build-x86-debug/ -name 'fold_relu_before_fq*.gcda'
build-x86-debug/src/vpux_compiler/CMakeFiles/npu_mlir_compiler_obj.dir/src/dialect/IE/transforms/passes/fold_relu_before_fq.cpp.gcda
```
4. Run gcov on the source code file. Explicitly specify respective object file:
```
gcov fold_relu_before_fq.cpp \
  --object-file \
  build-x86-debug/src/vpux_compiler/CMakeFiles/npu_mlir_compiler_obj.dir/src/dialect/IE/transforms/passes/fold_relu_before_fq.cpp.o
```
`gcov` is expected to print something similar to:
```
File 'propagate_quantize_dequantize.cpp
Lines executed:3.91% of 128
Creating 'propagate_quantize_dequantize.cpp.gcov'
...
```
Generated '*.gcov' files contain detailed information about code coverage.
