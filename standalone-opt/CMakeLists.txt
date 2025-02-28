set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# get_target_property(MLIRSupport_Sources MLIRSupport SOURCE_DIR)
# message(STATUS "sources of MLIRSupport=${MLIRSupport_Sources}")

add_llvm_executable(standalone-opt standalone-opt.cpp)
llvm_update_compile_flags(standalone-opt)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRParser
        MLIRPass
        MLIRStandalone
        MLIRSupport
        ast
        )

target_link_libraries(standalone-opt PRIVATE ${LIBS})
mlir_check_link_libraries(standalone-opt)
