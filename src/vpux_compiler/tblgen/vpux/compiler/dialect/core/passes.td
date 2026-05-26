//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_CORE_PASSES
#define VPUX_COMPILER_CORE_PASSES

include "mlir/Pass/PassBase.td"

//
// MoveDeclarationsToTop
//

def MoveDeclarationsToTop : PassBase<"move-declarations-to-top", "vpux::FunctionPass"> {
    let summary = "Reorder declaration operations to the beginning of their parent block";

    let description = [{
        AI-generated summary

        Reorders memory allocation and buffer declaration operations to the beginning of
        their parent block. Downstream passes (e.g., StaticAllocation, FeasibleAllocation)
        expect declarations to precede their uses within the same block.
    }];

    let constructor = "vpux::Core::createMoveDeclarationsToTopPass()";
}

//
// PrintDot
//

def PrintDot : PassBase<"print-dot", "vpux::ModulePass"> {
    let summary = "Convert current MLIR graph to Dot graph";

    let description = [{
        AI-generated summary

        Generates a Graphviz DOT representation of the IR operation graph and writes it to a
        file. Renders operations as nodes, def-use chains as edges, and region nesting as
        subgraphs, producing a visual graph that can be rendered with Graphviz dot for debugging
        and documentation.
    }];

    let options = [
        Option<
            "printOnlyDotInterfaceOpt", "print-only-dot-interfaces",
            "bool", "",
            "Print only ops that have dotInterface implementations"
        >,

        Option<
            "printDeclarationsOpt", "print-declarations",
            "bool", "",
            "Print declare memory operations"
        >,
        Option<
            "printConstOpt", "print-const",
            "bool", "",
            "Print const declare operations"
        >,
        Option<
            "printOnlyTaskAndBarrierOpt", "print-only-task-and-barrier",
            "bool", "",
            "Print only task and barrier operations"
        >,
        Option<
            "printOnlyAsyncExecOpt", "print-only-async-exec",
            "bool", "",
            "Print only async.exec operations"
        >,
        Option<
            "startAfterOpt", "start-after",
            "std::string", "",
            "Start graph generation after this operation"
        >,
        Option<
            "stopBeforeOpt", "stop-before",
            "std::string", "",
            "Stop graph generation after this operation"
        >,
        Option<
            "outputFileOpt", "output",
            "std::string", "",
            "Path to the output file"
        >,
        Option<
            "afterPassOpt", "pass",
            "std::string", "",
            "Print Dot after the pass"
        >
    ];

    let constructor = "vpux::Core::createPrintDotPass()";
}

//
// DumpIntermediateValues
//

def DumpIntermediateValues : PassBase<"dump-intermediate-values", "vpux::ModulePass"> {
    let summary = "Extract and dump intermediate operation values for debugging";

    let description = [{
        AI-generated summary

        Filters and dumps intermediate values from operations based on configurable filters
        specified via YAML configuration. Dumps selected operation results to files for
        offline comparison against reference values, typically used to locate the pass that
        introduces a numerical divergence.
    }];

    let options = [
        Option<
            "opFilters", "op-filters",
            "std::string", "",
            "Filters to select which operations to dump intermediate values from"
        >
    ];

    let constructor = "vpux::Core::createDumpIntermediateValuesPass()";
}

//
// PackNestedModules
//

def PackNestedModules : PassBase<"pack-nested-modules", "vpux::ModulePass"> {
    let summary = "Move functions into nested modules";

    let description = [{
        This pass collects all functions that sit in the module on which the pass is executed. It then performs an
        analysis to determine which functions can be nested inside sub modules. The pass has 2 modes: `default` and `entry-point`.
        When the `default` mode is on a function is nested in a submodule if
          - It does not contain a unit attribute named "do_not_nest"
          - It is not the network entry point
          - All its callees can be nested in the *same* submodule

        The third rule ensures that no nested function calls another function outside its own submodule as this is prohibited by
        `ModuleOp`'s `IsolatedFromAbove` property.
        This pass will try to separate nested functions into different submodules as much as possible.
        It will also add Attributes, PipelineOptions and Resources from the Main Module and NetworkInfoOp specific to the function
        present to subModules that contain a single functions.

        For example the following pseudo IR (`default` mode)

        ```MLIR
            module {
                func f()

                func g()

                func main() {
                    call @f()
                    call @g()
                }
            }
        ```

        would produce

        ```MLIR
            module {
                module @Module0 {
                    func f()
                }

                module @Module1 {
                    func g()
                }

                func main() {
                    Core.NestedCall @Module0::@f()
                    Core.NestedCall @Module1::@g()
                }
            }
        ```

        When the `entry-point` mode is on entryPoint function with all its callees are nested to
        one module called `NPUModule`.

        For example the following IR

        ```MLIR
            module @main_graph {
                net.NetworkInfo entryPoint @main
                func.func @func1() {...}
                func.func @func2() {...}
                func.func @main() {
                    call @func2
                }
            }
        ```

        would produce

        ```MLIR
            module @main_graph {
                func.func @func1() {...}
                module @NPUModule {
                    net.NetworkInfo entryPoint @main
                    func.func @func2() {...}
                    func.func @main() {
                        call @func2
                    }
                }
            }
        ```

        More involved examples can be found in the pass's LIT test files.

        Additionally, this pass supports explicit module packing via the `config.functionToPack` attribute.
        Functions marked with this attribute will be packed into a nested module with the specified name.
        If multiple functions have the same target module name, they will be grouped together.
        The `config.functionToPackEntryPoint` attribute can be used to mark a function as the entry point
        of the packed module.
        Note: Only direct children of the top module are processed; functions inside nested modules
        with the `config.functionToPack` attribute are not affected by this pass.
        Example:

        ```MLIR
            module {
                func.func private @helper() attributes {config.functionToPack = "my_module"} { ... }
                func.func private @entry() attributes {config.functionToPack = "my_module", config.functionToPackEntryPoint} { ... }
                func.func @main() {
                    call @entry()
                }
            }
        ```

        would produce

        ```MLIR
            module {
                module @my_module attributes {config.packedModule} {
                    func.func private @helper() { ... }
                    func.func private @entry() { ... }
                }
                func.func @main() {
                    Core.NestedCall @my_module::@entry()
                }
            }
        ```
    }];

    let options = [
        Option<
            "mode", "mode",
            "std::string", "\"default\"",
            "Nesting functions mode"
        >,
        Option<
            "enableProfiling", "enable-profiling",
            "bool", "false",
            "Enable profiling in nested modules' NetworkInfo"
        >
    ];

    let constructor = "vpux::Core::createPackNestedModulesPass()";
}


//
// UnpackNestedModules
//

def UnpackNestedModules : PassBase<"unpack-nested-modules", "vpux::ModulePass"> {
    let summary = "Eliminate any nested module, moving all functions into the main module";
    let constructor = "vpux::Core::createUnpackNestedModulesPass()";

    let description = [{
        This pass collects functions from any nested modules and moves them
        outside these modules into the main module. Any affected call-sites are
        updated accordingly. The pass has 2 modes: `default` and `entry-point`. With `default` mode all functions are recursively
        unpacked from nested modules and moved to the top one. Thus:

        ```MLIR
        module {
            module @initModule {
                func.func private @init() { ... }
            }
            func.func @main() {
                Core.Call @initModule::@init()
                ...
            }
        }
        ```

        Becomes:

        ```MLIR
        module {
            func.func private @init() { ... }
            func.func @main() {
                func.call @init()
                ...
            }
        }
        ````

        With `entry-point` mode the pass collects all functions and modules from a nested module (`NPUModule`), moves
        them to the top parent module and updates call ops. Thus:

        ```MLIR
            module @main_graph {
                func.func @func1() {
                    Core.Call @Module0::@func2
                    Core.Call @NPUModule::@nestedModule::func3
                }
                module @Module0 {
                    func.func @func2() {...}
                }
                module @NPUModule {
                    net.NetworkInfo entryPoint @main
                    module @nestedModule {
                        func.func @func3() {...}
                    }
                    func.func @main() {
                        Core.Call @nestedModule::@func3
                    }
                }
            }
        ```

        Becomes:

        ```MLIR
            module @main_graph {
                net.NetworkInfo entryPoint @main
                func.func @func1() {
                    Core.Call @Module0::@func2      // remains unchanged
                    Core.Call @nestedModule::func3  // changed
                }
                module @Module0 {
                    func.func @func2() {...}
                }
                module @nestedModule {
                    func.func @func3() {...}
                }
                func.func @main() {
                    Core.Call @nestedModule::@func3
                }
            }
        ```

        The pass also handles modules created via the `FunctionToPack` attribute. Modules marked with
        `config.packedModule = "FunctionToPack[...]"` attribute will be unpacked, and the original
        `config.functionToPack` attributes will be restored on the functions.
    }];

    let options = [
        Option<
            "mode", "mode",
            "std::string", "\"default\"",
            "Nesting functions mode"
        >
    ];
}

#endif
