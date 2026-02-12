# How to test

The project contains four types of tests:

- [unit tests](#unit-tests)
  - [GoogleTest based](#googletest-based-unit-tests)
  - [LLVM LIT based](#llvm-lit-based-unit-tests)
- [functional tests](#functional-tests)
- [end-to-end network validation](#end-to-end-network-validation)
- [fuzz tests](#fuzz-tests)
- [PSS validation](#pss-validation)

Most unit and functional tests make use of the GoogleTest framework. If you are unfamiliar with it, it is a good idea to explore the [primer](http://google.github.io/googletest/primer.html) it offers.

## Unit Tests

There are two types of unit tests present in the project:

* **GoogleTest** based
* **LLVM LIT** based

### GoogleTest based Unit Tests

The *GoogleTest* based unit tests can be found in the [tests/unit/](../tests/unit/) directory. Their main purpose is to validate the functionality of individual units of code.

#### How to run

To execute these tests, the `npuUnitTests` application can be used. It can be executed as a plain *GoogleTest* based application (including all command line options), without needing any specific environment setup.

The tests that are executed can be filtered using the `--gtest_filter` argument. A full list of all the available tests can be seen by using:

```sh
cd <openvino>/bin/<arch>/<build-type>
./npuUnitTests --gtest_list_tests
```

The full list of arguments can be seen by using `--help`.

#### How to add a new test

When adding a new component or altering the behavior of an existing one, it is usually a good idea to also introduce unit tests for it. These tests can be added in the [tests/unit/](../tests/unit/) directory, in a source file.

As detailed in the official GoogleTest documentation, tests can be:
- [simple ones](http://google.github.io/googletest/primer.html#simple-tests), where the test target is created along with the desired checks for its functionality;
- [test fixtures](http://google.github.io/googletest/primer.html#same-data-multiple-tests), when the test data and environment is set-up only once, over which multiple tests are executed;
- [parametrized](http://google.github.io/googletest/advanced.html#value-parameterized-tests), which is similar to test fixtures but the data received by the test is provided by its instantiations.

The type of test to utilize depends on the testing scope you want to introduce.

One example of a unit test is the following, found in [dims_order_tests.cpp](../tests/unit/vpux_compiler/core/attributes/dims_order_tests.cpp), which validates the functionality of the `DimsOrder::numDims()` method:

```C++
std::vector<std::pair<DimsOrder, size_t>> getOrders2Dims() {
    return std::vector<std::pair<DimsOrder, size_t>>{
            std::make_pair(vpux::DimsOrder::C, 1u),     std::make_pair(vpux::DimsOrder::NC, 2u),
            std::make_pair(vpux::DimsOrder::CHW, 3u),   std::make_pair(vpux::DimsOrder::HWC, 3u),
            std::make_pair(vpux::DimsOrder::HCW, 3u),   std::make_pair(vpux::DimsOrder::NCHW, 4u),
            std::make_pair(vpux::DimsOrder::NHWC, 4u),  std::make_pair(vpux::DimsOrder::NHCW, 4u),
            std::make_pair(vpux::DimsOrder::NCDHW, 5u), std::make_pair(vpux::DimsOrder::NDHWC, 5u)};
}

TEST_F(MLIR_DimsOrderTest, ValidateNumDimsTest) {
    auto orders2dims = getOrders2Dims();

    std::for_each(orders2dims.begin(), orders2dims.end(), [](const std::pair<DimsOrder, size_t>& order2dim) {
        EXPECT_EQ(order2dim.first.numDims(), order2dim.second);
    });
}
```

The test makes use of the [EXPECT_EQ](http://google.github.io/googletest/primer.html#assertions) macro provided by GoogleTest to compare the equality between the actual and expected values. Upon failure, a message is printed containing both values, the execution continues to compare the rest of the remaining pairs and the final test status is marked as failed. Other macros can also stop the execution on failure (i.e. `ASSERT_*`, such as `ASSERT_EQ`). Exceptions can also be caught using `EXPECT_ANY_THROW` / `ASSERT_ANY_THROW`. Please check the official documentation for the full list of features.

### LLVM LIT based Unit Tests

The *LLVM LIT* based unit tests make use of the [LLVM Integrated Tester](https://llvm.org/docs/CommandGuide/lit.html) framework, coupled with the [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) tool. Internally, they use Python scripts to run pattern-match-like tests. These tests requires Python 3 to be installed.

The tests can be found in the [tests/lit/NPU](../tests/lit/NPU) directory. They validate the functionality of the compiler, by executing passes or pipelines over input IRs and comparing the resulting IRs with the expected ones.

Since some of them are meant to be compatible with multiple architectures, they need to be parametrized with a specific architecture for each run. The architectural version is specified in the file name.

The tests are copied to the OpenVINO binary directory  (`<openvino>/bin/<arch>/<build-type>/lit-tests`) when the project is built.

#### How to run

To run the LIT tests, the following helper script can be used on Linux:

```sh
cd <openvino>/bin/<arch>/<build-type>/lit-tests
./run_all_lit_tests.sh
```

Or run the Python commands manually:

```sh
cd <openvino>/bin/<arch>/<build-type>/lit-tests
./lit-tool/lit.py --param arch=NPU37XX NPU/NPU
./lit-tool/lit.py --param arch=NPU40XX NPU/NPU
```

It is also possible to run a specific LIT test using the `lit.py` tool manually. For example:

```sh
cd <openvino>/bin/<arch>/<build-type>/lit-tests
# The `-a` argument is useful for seeing the command executed for the test and the error message, if one is present
./lit-tool/lit.py -a --param arch=NPU40XX NPU/NPU/dialect/VPU/passes/init_compiler_40XX.mlir
```

Manually running the command specific to a test is also useful. The command can be found in the `RUN` comment at the start of each test file. For example:

```sh
cd <openvino>/bin/<arch>/<build-type>/lit-tests
../vpux-opt --init-compiler="vpu-arch=NPU40XX compilation-mode=ReferenceSW" NPU/NPU/dialect/VPU/passes/init_compiler_40XX.mlir | ../FileCheck NPU/NPU/dialect/VPU/passes/init_compiler_40XX.mlir --strict-whitespace
```

Using this manual command gives you more control, as you can change the parameters for the execution or omit the `FileCheck` command in order to see the resulting IR.

**Note:** In order to run the LIT tests on updated test sources, the project has to be rebuilt in order to copy the updated test sources from the compiler into OpenVINO binary directory. However, if you have used a developer preset when building the project, `LIT_TESTS_USE_LINKS` will be enabled which will remove the need to build the project, as symbolic links will be used to point to the updated sources.

#### How to add a new test

Tests are added in `.mlir` files inside the [tests/lit/NPU](../tests/lit/NPU) base directory. Some of the targets that LIT tests can cover are:

- passes
- pipelines
- operation canonicalizers & folders
- frontend & backend translation

The test file name has to match the pass name. For example, the test file name for the  `OptimizeCopiesPass` should be `optimize_copies.mlir`. If necessary, it is also possible to divide the file into several specialized test suites and use a suffix to differentiate them: `optimize_copies_ddr.mlir`, `optimize_copies_cmx.mlir`. Some extra considerations:

- The tests which are intended for all architectures should follow the general naming rules, otherwise a suffix should be added to mark the supported device (those listed in the `REQUIRES` field): `optimize_copies_37XX_40XX.mlir`.
- Also use postfixes such as `40XX+` to denote that this test is ran for all platforms since `NPU4` (ex: `test_40XX+.mlir`); with the mention that whenever there's a drop in that feature (for example in platforms that are newer than NPU5), the test should be renamed to explicitly denote all the platforms (ex: `test_40XX_50XX.mlir`).
- Every test needs the `REQUIRES` field to work correctly.

##### RUN command

What the tests in a given `.mlir` file target depends on the `RUN` command found at the start of the file. This command accepts [patterns](https://llvm.org/docs/CommandGuide/lit.html#substitutions) that can substituted in order to obtain the final executable shell command. The most common pattern is `%s`, which will be replaced with the path to the `.mlir` test file. This allows us to employ `vpux-opt` in order to execute passes, pipelines, canonicalizers etc. over IRs and validate the result with FileCheck. For example:

```MLIR
// RUN: vpux-opt --init-compiler %s | FileCheck %s
```

The result of the substitution can be found when running `lit.py -a` over the test file.

The `RUN` command of a test may be parametrized, for example with `%arch%`. The value of this parameter is controlled using the `--param arch` argument of `lit.py`. The accepted values for this parameter can also be controlled for a particular test instance using the `REQUIRES` comment. If the value of the `%arch%` parameter does not match with the condition in the `REQUIRES` field, the test run is skipped and marked as UNSUPPORTED. Here is an example of how a parameter can be specified and filtered in a test:

```MLIR
// RUN: vpux-opt ... %arch% ...
// REQUIRES: arch-NPU37XX || arch-NPU40XX
```

The device type must always be specified in order for the appropriate passes to be registered. There are two ways to do this:

1. Using the `--vpu-arch` command-line argument. Used only to test pipelines (e.g. `DefaultHW`, `ReferenceSW`, etc.):
    ```MLIR
    // RUN: vpux-opt --vpu-arch=NPU40XX --split-input-file --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode %s | FileCheck %s
    ```

    - `vpux-arch` is also required to be used with `vpux-translate` to specify the platform for import:
    ```sh
    ./vpux-translate --vpu-arch=NPU40XX --import-IE <xml path> --mlir-print-debuginfo -o net.mlir
    ```
    and export:
    ```MLIR
    // RUN: vpux-opt --init-compiler="vpu-arch=NPU40XX" %s | vpux-translate --vpu-arch=NPU40XX --export-VPUIP -o %t
    ```

    > Note: TODO(E#84874): currently, `--vpu-arch` is used for both import and export. However, it would be a better option to extract arch info from module for the export case.


2. Using the `--init-compiler` pass, which accepts `vpu-arch` as an option. Used to test passes and sub-pipelines:
    ```MLIR
    // RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=NPU40XX" --unroll-distributed-ops  %s | FileCheck %s
    ```

    In some cases, it is necessary to set custom resources in the IR:
    ```MLIR
    module @memory {
        config.MemoryResource 10000 bytes of @CMX_NN
    }
    ```

    There is a special option `allow-custom-values` which sets the remaining unspecified resources and leaves the existing ones untouched:
    ```MLIR
    // RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=NPU40XX allow-custom-values=true" --unroll-distributed-ops  %s | FileCheck %s
    ```

> Note: there is a plan to change the two options, so that only one is used: E#82305

##### Content

Beside the execution commands mentioned above, the test `.mlir` files also contain the input IRs and the expected results after the execution of the `RUN` command.

Let's see this in practice with an example, using a test found in [split_fake_quant.mlir](../tests/lit/NPU/dialect/IE/passes/split_fake_quant.mlir):

```MLIR
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-fake-quant %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>
// CHECK: !qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL:  func.func @SingleQuantParams
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @SingleQuantParams(%input: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %fq = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %fq : tensor<1x3x30x30xf32>

    // CHECK:       [[QUANT:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[DEQUANT:%.*]] = IE.Dequantize([[QUANT]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[DEQUANT]]
}
```

The test's `RUN` command executes the `vpux-opt` tool with the following arguments:

- `--split-input-file`: separates the content of the file into separate executions / tests, which are delimited by `// -----`. All of the sections found between these delimiters are treated as if they came from separate files by `vpux-opt`, so that the elements defined in previous sections do not affect the behavior of the current test. If more tests are added in a file, it is recommended to delimit them and use `--split-input-file`.
- `--init-compiler="vpu-arch=%arch%"`: executes the InitCompiler pipeline over an input IR, while also setting the `vpu-arch` pass option to the value of the `%arg%` parameter (which can only take values `arch-NPU37XX` or `VPUX40XX`, as seen in the `REQUIRES` command).
- `--split-fake-quant`: executes the SplitFakeQuant pass over the IR resulted after the InitCompiler pipeline.

After `vpux-opt` finishes execution, the resulting IR is printed. To ensure that the resulting IR looks as expected, it is fed to `FileCheck`. This tool in turn will try to match all of the checks from the test file (e.g. `CHECK`, `CHECK-LABEL`, `CHECK-SAME`) with the IR and report any discrepancies. That is why it receives both the result from `vpux-opt` with stdin and the path to the `.mlir` test file (i.e. `%s`). The tool will compare its input IR with the checks in order to find the first match. If a string is not found, an error is reported with some potential matches that look very similar to the intended string. After the first match has been found in the IR, the string in the next check instruction is searched, starting from the end of the last match. This is repeated until all checks are matched.

Some of the more essential checks offered by FileCheck are the following (examples can be found [here](https://llvm.org/docs/CommandGuide/FileCheck.html)):
- `CHECK`: the exact string must be matched in the IR
- `CHECK-SAME`: similar with `CHECK`, but the match has to take place on the same line, after the last match
- `CHECK-NEXT`: similar with `CHECK`, but the match has to take place on the next line
- `CHECK-NOT`: the string must not exist in the IR, between the previous and next match (or before the first match or after the last match, depending on the positioning of this check)
- `CHECK-DAG`: the strings might not be placed in the IR in the exact order of the checks, but all `CHECK-DAG`s must be matched in the end
- `CHECK-LABEL`: matches labels or unique identifiers; it behaves similarly to `CHECK`, but it is meant to capture labels or other unique identifiers
- `{LITERAL}`: modifier to perform a literal match (e.g. by treating regular expressions as normal strings); can be added to the checks above (e.g. `CHECK{LITERAL}`)

FileCheck also supports regex, by capturing the expression between `{{` and `}}`. Alternatively, a string or expression can also be captured and substituted across checks, by using `[[NAME:expression]]` for capturing and `[[NAME]]` for substituting the same captured expression. For more in-depth information on how FileCheck works, please look over the [official documentation](https://llvm.org/docs/CommandGuide/FileCheck.html).

There are a few different ways to generate input IRs for the test target:

- manually, which can be preferred for small test cases;
- by compiling a relevant network, dumping the IR before the target executes and extracting the relevant parts of the IR (see [how_to_debug.md](../src/vpux_compiler/docs/guides/how_to_debug.md) for instructions on dumping the IR).

Additionally, LIT can support test scenarios which are expected to fail. This can be done using the [XFAIL](https://llvm.org/docs/TestingGuide.html#constraining-test-execution) feature, which can expect failure every time or only based on some constraints. Tests can also be disabled by using `REQUIRES: DISABLED`.

##### Guidelines

1. To determine what a test suite should target, it is generally a good idea to consider all of the relevant scenarios for the particular target. None of the target code should remain untested.
    - That being said, the tests constructed for one target should be kept simple, so that they mostly contain relevant scenarios. For example, adding test cases with float and quantized types when a target is data type invariant is likely redundant. Similarly, it is recommended to keep a test as small as possible while still providing reasonable coverage. One example: testing an operation that must take input from CMX. There's no need for `DDR -> CMX` DMA operations in the IR, since you can create a synthetic test where the data resides in CMX from the beginning.
    - For the example above, the behavior of the `SplitFakeQuant` pass was validated when it split an IE.FakeQuantize operation into two other operations. Other instances can also be created, such as when the quantization is done on an axis or when the splitting cannot take place.

2. It is generally a good idea for the checks to cover the essential changes that are expected to be done by the target, rather than matching the entire IR.

3. It is recommended for all tests to first execute the `InitCompiler` pipeline (if possible), so that the information about the architecture is added into the IR. This information can influence the behavior of the real target of the test, so it should always be present.

4. All values should be captured and substituted instead of used directly as they appear in the resulting IR. In other words, values such as `%0`, `%cst` or `%arg0` should be captured where they are defined (e.g. `[[CST:%.+]]`) and substituted wherever they are used (e.g. `[[CST]]`).

    ```MLIR
    // OK
    // CHECK-LABEL:  func.func @test
    // CHECK-SAME:     ([[INPUT:%.+]]: tensor<10xf16>)
    func.func @test(%input: tensor<10xf16>) -> ... {
        %cst = const.Declare ...
        %reshape = IE.Reshape(%input) ...
        %add = IE.Add(%input, %reshape) ...
        return %add

        // CHECK:  [[CST:%.+]] = const.Declare ...
        // CHECK:  [[RESHAPE:%.+]] = IE.Reshape([[CST]]) ...
        // CHECK:  [[ADD:%.+]] = IE.Add([[INPUT]], [[RESHAPE]]) ...
        // CHECK:  return [[ADD]]
    }

    // BAD: Using values directly is not safe because the names of the values can change,
    //      which will result in the need of refactoring all of the checks. For example,
    //      when slightly altering the IR (e.g. removing an operation), when the operation
    //      order in the IR can vary (e.g. when `CHECK-DAG` needs to be used) or even
    //      potentially when LLVM is upgraded
    func.func @test(%arg0: tensor<10xf16>) -> ... {
        %0 = const.Declare ...
        %1 = IE.Reshape(%0) ...
        %2 = IE.Add(%arg0, %1) ...
        return %2

        // CHECK:  %0 = const.Declare ...
        // CHECK:  %1 = IE.Reshape(%0) ...
        // CHECK:  %2 = IE.Add(%arg0, %1) ...
        // CHECK:  return %2
    }
    ```

5. When capturing values, the `.+` expression is preferred to be used over `.*`:

    ```MLIR
    // OK
    // CHECK:  [[CST:%.+]] = const.Declare tensor<1x8x4x4xf32>

    // BAD: The asterisk indicates zero or more occurrences of the preceding element.
    //      But at least one character must be present
    // CHECK:  [[CST:%.*]] = const.Declare tensor<1x8x4x4xf32>
    ```

6. The `%` character used in the name of the values should be captured as well:

    ```MLIR
    // OK
    // CHECK:      [[CST:%.+]] = const.Declare tensor<1x8x4x4xf32> = dense<5.000000e+00> : tensor<1x8x4x4xf32>
    // CHECK-NOT:  const.Declare
    // CHECK-NOT:  IE.Add
    // CHECK:      return [[CST]]

    // BAD: You need to write `%` again when using the captured variable
    // CHECK:      %[[CST:.+]] = const.Declare tensor<1x8x4x4xf32> = dense<5.000000e+00> : tensor<1x8x4x4xf32>
    // CHECK-NOT:  const.Declare
    // CHECK-NOT:  IE.Add
    // CHECK:      return %[[CST]]
    ```

7. Each value, both from the input IR and from the checks, should have a meaningful name:

    ```MLIR
    // OK
    // CHECK-DAG:  [[FILTER:%.+]] = const.Declare tensor<16x3x3x3xf32> = dense<1.000000e+00> : tensor<16x3x3x3xf32>
    // CHECK-DAG:  [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK:      [[CONV:%.+]] = IE.Convolution([[ARG0]], [[FILTER]], [[BIAS]])
    // CHECK:      return [[CONV]]

    // BAD: Such a test is much more difficult to understand,
    //      since the origin of the variables and what they mean are unknown.
    //      There is also a high probability of making a "green" test for the wrong behavior of the pass
    // CHECK-DAG:  [[VAR1:%.+]] = const.Declare tensor<16x3x3x3xf32> = dense<1.000000e+00> : tensor<16x3x3x3xf32>
    // CHECK-DAG:  [[VAR2:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK:      [[VAR3:%.+]] = IE.Convolution([[VAR0]], [[VAR1]], [[VAR2]])
    // CHECK:      return [[VAR4]]
    ```

8. The tests should be easy to read. For this purpose, both the input IR and the checks should be split across multiple lines (e.g. using `CHECK-SAME`) when necessary. Aligning the checks vertically can also help with readability.

    ```MLIR
    // OK
    // CHECK:       [[CONV:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      kernel_size = [1, 1]
    // CHECK-SAME:      kernel_strides = [1, 1]
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>

    // BAD: It is unnecessarily difficult to understand what the resulting IR is expected to contain
    // CHECK: [[CONV:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
    ```

## Functional tests

The main purpose of these tests is to validate the functionality of the project. This is mainly done by compiling single or multi-layer networks, running inferences and comparing the results with a reference, usually obtained from the CPU plugin.

Functional tests can be found in the the [tests/functional/](../tests/functional/) directory. They are also based on the GoogleTest framework, similar to the unit tests mentioned above.

#### How to run

To execute these tests, the `npuFuncTests` application can be used. The full list of tests can be found using:

```sh
cd <openvino>/bin/<arch>/<build-type>
# Note: The platforms always has to be specified when executing `npuFuncTests`. One way this can be done is via the `IE_NPU_TESTS_PLATFORM` environment variable.
#       This variable has no impact over what tests get listed, as all of them will be printed.
IE_NPU_TESTS_PLATFORM=NPU4000 ./npuFuncTests --gtest_list_tests
```

To only run some of the tests, the `--gtest_filter` argument can be used. The full list of arguments can be seen by using `--help`.

After deciding what tests to run, it is necessary to select which plugin to use for the inference. NPU plugin is generally only usable if you have an NPU driver installed, which usually means that you have a CiD build and an NPU on your device. The plugin can be selected with the `-d` argument: `-d NPU`. If this argument is not specified, NPU plugin will be used.

`npuFuncTests` also provides multiple features which are useful during development. They are exposed via environment variables:

- `IE_NPU_TESTS_RUN_INFER`: allows you to only test the compilation and skip the inferences (i.e. skip the accuracy testing)
    - `IE_NPU_TESTS_RUN_INFER=1` (default): run the inference stage of the tests
    - `IE_NPU_TESTS_RUN_INFER=0`: skip the inferences

- `IE_NPU_TESTS_DUMP_PATH`: allows you to dump artifacts generated during the test execution (e.g. the compiled blob, input binaries, results etc.)
    - example: `IE_NPU_TESTS_DUMP_PATH=artifacts` - the files will be stored in a directory called `artifacts`, found in the current working directory
    - This variable needs to be used in conjunction with others, which will select what artifacts should be dumped:
        - `IE_NPU_TESTS_EXPORT_INPUT=1`: export the input binaries that are used to validate the accuracy of the test case
        - `IE_NPU_TESTS_EXPORT_OUTPUT=1`: export the output binaries that are obtained from running the inference on NPU
        - `IE_NPU_TESTS_EXPORT_REF=1`: export the reference output binaries that are used to compare the NPU results against
        - `IE_NPU_TESTS_RUN_EXPORT=1`: export the compiled blob

- `IE_NPU_TESTS_IMPORT_REF`: allows you to use custom references binaries instead of those provided / generated by the test
    - `IE_NPU_TESTS_IMPORT_REF=0` (default): use the references provided by the test
    - `IE_NPU_TESTS_IMPORT_REF=1`: use custom reference binaries; the names of these binaries are unique per test case and must be the same as those exported by `IE_NPU_TESTS_EXPORT_REF`

- `IE_NPU_TESTS_IMPORT_INPUT`: allows you to use custom input binaries instead of those provided / generated by the test
    - `IE_NPU_TESTS_IMPORT_INPUT=0` (default): use the input binaries provided by the test
    - `IE_NPU_TESTS_IMPORT_INPUT=1`: use custom input binaries; the names of these binaries are unique per test case and must be the same as those exported by `IE_NPU_TESTS_EXPORT_INPUT`

- `IE_NPU_TESTS_LONG_FILE_NAME`: allows longer file names for the exported artifacts; by default shorter file names are used for all operating systems

Additionally, there are some environment variables specific to the platform / device selection:

- `IE_NPU_TESTS_PLATFORM`: sets the value for the `NPU_PLATFORM` property (sample value: `NPU4000`)
- `IE_NPU_TESTS_DEVICE_NAME`: passed to OpenVINO as the name of the device for loading the appropriate plugin; default: `NPU`

#### How to add a new test

The methodology of adding a new functional test is similar to the one for unit tests, as it uses GoogleTest. However, there are some helper classes which can help simplify the process for some scenarios. For example, adding a new single-layer test can be done by employing the layer-specific classes provided by OpenVINO. An example can be found in [batch_norm.cpp](../tests/functional/shared_tests_instances/single_layer_tests/batch_norm.cpp):

```C++
#include "single_op_tests/batch_norm.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class BatchNormLayerTestCommon : public BatchNormLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(BatchNormLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<std::vector<ov::Shape>> inShapes = {
        std::vector<ov::Shape>{{1, 5, 20, 20}},
};
const auto params = testing::Combine(testing::Values(0.001),             // epsilon
                                     testing::Values(ov::element::f16),  // netPrc
                                     testing::ValuesIn(static_shapes_to_test_representation(inShapes)),
                                     testing::Values(test_utils::TARGET_DEVICE));
INSTANTIATE_TEST_SUITE_P(precommit_BatchNorm, BatchNormLayerTestCommon, params,
                         BatchNormLayerTestCommon::getTestCaseName);

}
```

The test uses the `BatchNormLayerTest` and `VpuOv2LayerTest` base classes, letting the test user focus on the parameters of the layer instead (e.g. input shape, layout etc). These base classes handle the creation of the network behind the scenes (containing this single layer), the compilation of the network into a blob using our compiler, the execution of an inference and comparison of the result against a reference. The reference is obtained on the same network, using the same input, by running an inference with the CPU plugin.

All of the single-layer tests can be found in the [tests/functional/shared_tests_instances/single_layer_tests/](../tests/functional/shared_tests_instances/single_layer_tests/) directory.

Besides these tests, more complex compilation scenarios can also be tested by using subgraphs. Such tests can be found in the [tests/functional/subgraph_tests/](../tests/functional/subgraph_tests/) directory. They can also make use of the `VpuOv2LayerTest` helper class, but the creation of the network has to be defined manually, in the `SetUp()` method of the base test class.

#### How to add a skip filter

Skip filters are used to select which tests can run on specific devices, backends or operating systems.
By default `npuFuncTests` and `ov_npu_func_tests` do not have any skips configured, to enable skips it is necessary to set an environment variable with the path to the skip config file.

By default, the environment variable `OV_NPU_TESTS_SKIP_CONFIG_FILE` is set find `skip_tests.xml` in the current working folder. `OV_NPU_TESTS_SKIP_CONFIG_FILE` has to be set with a valid path to an `.xml` file containing filters with the following structure:

```xml
<skip_configs>
    <skip_config>
        <message>skip_message_xxxxxx</message>
        <enable_rules>
            <backend>LEVEL0</backend>
            <backend></backend> (special case for no backend)
            <device>3720</device>
            <device>!4000</device> (using "!" to negate rule)
            <operating_system>windows</operating_system>
            <operating_system>linux</operating_system>
        </enable_rules>
        <filters>
            <filter>skip_filter_xxxxxxxxxx</filter>
            <filter>skip_filter_xxxxxxxxxx</filter>
            <filter>skip_filter_xxxxxxxxxx</filter>
        </filters>
    </skip_config>
</skip_configs>
```

Skip filters can be enabled/disabled according to rules defining the device, backend or operating system, depending on where tests are supposed to run. Rules are optional and users can negate a rule by using `!`, they can add as many rules as needed. When evaluating if a skip filter is enabled, different rule categories (backend, device, operating_system) use an `AND` operation between each other. While multiple entries of the same category will use an `OR` operation.

## Fuzz tests

Fuzz testing is also used in the project, in order to identify potential issues related to undefined behavior, invalid memory accesses etc.

These tests can be found in the the [tests/fuzz/](../tests/fuzz/) directory. Detailed information on these tests can be found in the [README](../tests/fuzz/README.md) file.
