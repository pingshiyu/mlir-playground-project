set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# get_target_property(MLIRSupport_Sources MLIRSupport SOURCE_DIR)
# message(STATUS "sources of MLIRSupport=${MLIRSupport_Sources}")

# Build 2 binaries: one for optimisation within MLIR, the other to also compile .test scripts.
add_llvm_executable(test-compiler test-compiler.cpp)
llvm_update_compile_flags(test-compiler)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRParser
        MLIRPass
        MLIRStandalone
        MLIRSupport
        MLIRAnalysis
        MLIRIR
        MLIRSideEffectInterfaces
        MLIRTransforms
        MLIRTest
        ast
        )

target_link_libraries(test-compiler PRIVATE ${LIBS})
mlir_check_link_libraries(test-compiler)
