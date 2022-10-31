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
}

#endif // TEST_DIALET