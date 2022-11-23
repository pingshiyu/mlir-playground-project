#ifndef TEST_DIALECT
#define TEST_DIALECT

include "mlir/IR/OpBase.td"


def Test_Dialect : Dialect {
    let name = "test"; // dialect namespace 
    let cppNamespace = "::mlir::test"; // cpp dialect namespace
    let summary = "A test dialect, following Toy tutorial on MLIR page, and the Standalone"
                  "template on llvm/mlir repo.";
    let description = [{
        This is really just a dialect to get myself familiar with working with 1. CMake, and 
        LLVM's tooling around it, 2. feel for MLIR's structure and philosophies.
    }];

    // We set this bit to generate a declaration of the `materializeConstant`
    // method so that we can materialize constants for our toy operations.
    let hasConstantMaterializer = 1;

    // We set this bit to generate the declarations for the dialect's type parsing
    // and printing hooks.
    let useDefaultTypePrinterParser = 1;
}

#endif // TEST_DIALET