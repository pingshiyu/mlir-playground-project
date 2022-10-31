# Scratchpad for exploring MLIR build/compilation structure
This is going to be quite messy. Using this repo to learn:
- CMake
- LLVM CMake
- Building & writing MLIR dialects
- MLIR semantics

# Notes
## What is MLIR?
MLIR is a convenient way to specify language constructs, and associated optimisations / verification on those constructs. By constructs, we mean the syntax of the language. This syntax is represented as C++ objects. 

With the syntax, the framework provides easy ways to construct tooling / useful functionalities around it. These can include things like type checking (verifier), rewrites (optimisations), or printers.

## What doesn't MLIR provide?
Missing from a full compiler are these components:
- Lexer & parser into ASTs, or anyhow a way to obtain/reach these MLIR objects we defined to represent the language (but LLVM has tools for building this) *Question: what do these MLIR objects look like? E.g. how do they compose?*
- Custom implementation for generated code blobs. E.g. specific type verifiers, custom factories (builders). *Question: when will these verifiers be called?*
- 

The above code will need to be written separately.


## Keywords & concepts
* MLIRGen: conventional name, file used for generating MLIR objects from ASTs
* Op vs Operation class: Op has a pointer to an Operation*. The Operation object contains the common methods and attributes shared by all operations, and Op contains the operation-specific information. One can always cast between Op and Operation* types.
    * Outstanding: Adaptors in Op classes, what does it to?
* build: factory methods for constructing Op objects
* assemblyFormat: specify our own way to print out operations

# An out-of-tree dialect template for MLIR

This repository contains a template for an out-of-tree [MLIR](https://mlir.llvm.org/) dialect as well as a
standalone `opt`-like tool to operate on that dialect.

## How to build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-standalone-opt
```
To build the documentation from the TableGen description of the dialect
operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.

## License

This dialect template is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.
