# actually produces the Standalone library (MLIRStandalone)
# baking a binary with the implementation cpps, and including the headers they pull in
add_mlir_dialect_library(MLIRStandalone
        StandaloneDialect.cpp
        StandaloneOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        MLIRStandaloneOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
