# actually produces the Standalone library (MLIRStandalone)

# add the optimisastions patterns file
set(LLVM_TARGET_DEFINITIONS Optimisations.td)
mlir_tablegen(Optimisations.inc -gen-rewriters)
add_public_tablegen_target(TestOptimisations)

# baking a binary with the implementation cpps, and including the headers they pull in
add_mlir_dialect_library(MLIRTest
        TestOps.cpp
        TestDialect.cpp
        MLIRGen.cpp
        Optimisations.cpp
        Interfaces.cpp
        Passes.cpp
        LowerToAffineLoops.cpp
        LowerToLLVM.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/TestDialect

        DEPENDS
        MLIRTestOpsIncGen
        TestOptimisations

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
