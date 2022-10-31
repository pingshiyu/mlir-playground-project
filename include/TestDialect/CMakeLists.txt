# translate the .td files
# include the header files
add_mlir_dialect(TestOps test)
add_mlir_doc(TestDialect TestDialect TestDialect/ -gen-dialect-doc)
add_mlir_doc(TestOps TestOps TestDialect/ -gen-op-doc)