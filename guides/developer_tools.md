# Developer tools

This document describes a list of tools that are useful during development. These are tools that automate the detection of coding style violations, can identify problematic code or simply speed up development. If you intend to contribute to the project, it is recommended to install them locally.

## Pre-commit hooks

The project uses multiple git hooks for ensuring the code formatting. The [pre-commit framework](https://pre-commit.com/) is used for managing the hooks. The recommended way of installing it is using `pip`:
```sh
$ pip install pre-commit
```

The hooks can then be installed into your local clone using:
```sh
$ pre-commit install
```

When the hooks are executed for the first time, their dependencies will automatically be installed. For example, when making a commit using the `git commit` command. The hooks can also be executed manually over some explicit files or over all the files in the project:
```sh
# Run all hooks over all files
$ pre-commit run --all-files

# Run the clang-format hook over some files, over all files or over a set of commits
$ pre-commit run clang-format --files PATH/TO/FILES
$ pre-commit run clang-format --all-files
$ pre-commit run clang-format --from-ref FROM_REF --to-ref TO_REF
```

Note that the versions of the tools used by the hooks are tracked in the [.pre-commit-config.yaml](../.pre-commit-config.yaml) file.

Uninstalling the hooks can be done using `pre-commit uninstall`.

### Skipping the hooks

If a commit is erroneously prevented from being created by a hook, that hook can be skipped by adding its ID in the `SKIP` environmental variable; multiple hooks can be skipped by separating them via commas. Alternatively, all hooks can be skipped by using the `--no-verify` argument.

```sh
SKIP=hook-id git commit
SKIP=hook1-id,hook2-id git commit
git commit --no-verify
```

## Ninja

[Ninja](https://ninja-build.org/) is a build generator with a focus on speed as an alternative to default: Unix Makefiles and Visual Studio. It is used by all of the instructions in the [how_to_build.md](./how_to_build.md) document.

On Ubuntu, Ninja can be installed via standard packaging system

```sh
sudo apt install ninja-build
```

On Windows, the recommended way is to install it via `Visual Studio Installer`. The steps are:

1. Open `Visual Studio Installer`
2. `Modify` and then `Individual components`
3. Search for `C++ CMake tools for Windows`
4. Check the component and install it

If you cannot follow the recommended ways, you can find a way which works for you by checking out the "Getting Ninja" section on the [official website](https://ninja-build.org/).

## Ccache

[Ccache](https://ccache.dev/) is a compiler cache, which speeds up recompilation by caching the result of previous compilations and detecting when the same compilation is being done again.

On Ubuntu, Ccache can be installed via standard packaging system:

```sh
sudo apt install ccache
```

On Windows, it can be installed as follows:

1. Download latest version of [Ccache binaries](https://github.com/ccache/ccache/releases) or build from source code
2. Add path to `ccache.exe` file into `PATH` environment variable.
3. Set the cache size to 10-20+ GB (default 5 GB is not enough for OpenVINO + compiler build):
```sh
ccache -M 20 GB
```
4. When building the project via developer presets, Ccache will be used automatically. If building manually, you can manually add it to the project by setting the following CMake variables:
```sh
"CMAKE_C_COMPILER_LAUNCHER":"ccache"
"CMAKE_CXX_COMPILER_LAUNCHER":"ccache"
```
5. Check the Ccache statistics during / after build, to make sure the cache is being used. Example of `ccache -s` output:
```
Cacheable calls:    8980 / 23702 (37.89%)
  Hits:             1198 /  8980 (13.34%)
    Direct:         1176 /  1198 (98.16%)
    Preprocessed:     22 /  1198 ( 1.84%)
  Misses:           7782 /  8980 (86.66%)
Uncacheable calls: 14722 / 23702 (62.11%)
Local storage:
  Cache size (GB):  1.55 / 20.00 ( 7.73%)
```

## clang-tidy

[clang-tidy](https://clang.llvm.org/extra/clang-tidy/) is a static analyzer which can detect coding style violations and common issues.

On Ubuntu, clang-tidy can be installed via standard packaging system:

```sh
sudo apt install clang-tidy
```

Alternatively, the desired version can be installed via LLVM's [apt repository](https://apt.llvm.org/).

One way to utilize clang-tidy is via your IDE. For example, if you are using Visual Studio Code, you can install [clangd](https://clangd.llvm.org/installation) and the associated Visual Studio Code [plugin](https://clangd.llvm.org/installation#editor-plugins), [vscode-clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd). Please note that [compile_commands.json](https://clangd.llvm.org/installation#project-setup) needs to be created for your build in order for clangd to work; the developer presets mentioned in the [how_to_build.md](./how_to_build.md) document will ensure this file gets created, but you can also enable it for your manual build by setting the following build option:
```sh
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1
```

The developer presets will place the `compile_commands.json` file inside the `build-x86_64/<build_type>` directory. For clangd to find it, you may also need to specify the path to it in your IDE. For example, in Visual Studio Code, you can specify it in your workspace by adding a `.clangd` file to the root of the repository which contains:

```
CompileFlags:
  CompilationDatabase:
    ./<build-dir>

Diagnostics:
  ClangTidy:
    FastCheckFilter: None
  MissingIncludes: Strict
```

- `CompilationDatabase`: points to the build directory that contains the `compile_commands.json` file (for example, `./build-x86_64/Debug`)
- `FastCheckFilter` (optional): disables filtering of the checks of clang-tidy, even if they're treated as "not fast"
- `MissingIncludes` (optional): enables `IncludeCleaner` to complain about missing includes (everything used in a file should be included directly)

> Note: As compile commands database is used by clangd to get metadata about codebase (which files are used for build and with which options), files that aren't listed there aren't supported by clangd. In our case it means MLIR auto-generated files (*.hpp.inc, *.cpp.inc) won't be fully supported; they aren't listed in compile commands database because they are basically just text (code without needed includes) and not compiled directly.

> Note: If you are using the C/C++ extension in Visual Studio Code along clangd, you need to disable its IntelliSense feature. A notification should appear automatically to disable this feature; if it does not appear, it can be manually disabled in the extension's settings.
