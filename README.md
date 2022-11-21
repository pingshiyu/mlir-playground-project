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
    - *Question: what does the default verifiers do?*

## What doesn't MLIR provide?
Missing from a full compiler are these components:
- Lexer & parser into ASTs, or anyhow a way to obtain/reach these MLIR objects we defined to represent the language (but LLVM has tools for building this) 
    - *Question: what do these MLIR objects look like? E.g. how do they compose?*
    - *Answer:* Operations are the base unit for organisation. Roughly: Operations has *Regions*, which are just containers for *List Blocks*. These *Blocks* contains lists of *Operations* again. Can view *Operations* as a function + its body, *Regions* as the function body container, and *Blocks* as the function body.
- Custom implementation for generated code blobs. E.g. specific type verifiers, custom factories (builders).
    - *Question: when will these verifiers be called?* 

The above code will need to be written separately.

## Keywords & concepts
* MLIRGen: conventional name, file used for generating MLIR objects from ASTs
* Op vs Operation class: Op has a pointer to an Operation*. The Operation object contains the common methods and attributes shared by all operations, and Op contains the operation-specific information. One can always cast between Op and Operation* types.
    * Outstanding: Adaptors in Op classes, what does it to?
        * Answer: These adaptors are used in the Op classes implementations.
* build: factory methods for constructing Op objects
* hasCustomAssemblyFormat: specify our own way to print out operations. This is cosmetic. Few options here:
    1. Nothing: just use the default MLIR object printing method
    2. let assemblyFormat = `format`: `format` gives a shorthand to custom pretty print your operation
    3. hasCustomAssemblyFormat = 1: you need to specify the Op::print and Op::parse functions in the .cpp
* parser: its job is to parse MLIR-style strings back into MLIR objects. Parsers for custom formats can be specified by the DSL if it is sufficiently simple. Otherwise custom code can be written too for the parser. 
* Operand: operation arguments which are produced at runtime by other operations.
* Attributes: operation arguments which are compile time constant
* Rewrites: Rewrites can be requested by setting `let hasCanonicalizer = 1;` in the Op definition. This allows it to register rewrites (which are classes inheriting the `: mlir::OpRewritePattern<OpType>` interface, to be implemented), and these will be called later on by a `PassManager`, applying these rewrites in a processes called `Canonicalization`.

# An out-of-tree dialect template for MLIR

This repository contains a template for an out-of-tree [MLIR](https://mlir.llvm.org/) dialect as well as a
standalone `opt`-like tool to operate on that dialect.

## How to build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit -DUSE_SANITIZER="Address;Undefined"
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
