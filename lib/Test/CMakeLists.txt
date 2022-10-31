# actually produces the Standalone library (MLIRStandalone)
# baking a binary with the implementation cpps, and including the headers they pull in
add_mlir_dialect_library(MLIRTest
        TestOps.cpp
        TestDialect.cpp
        MLIRGen.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/TestDialect

        DEPENDS
        MLIRTestOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
