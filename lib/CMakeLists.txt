# Enable errors for any global constructors.
add_flag_if_supported("-Werror=global-constructors" WERROR_GLOBAL_CONSTRUCTOR)

# compiles the various libs we need. including the dialect library binaries
add_subdirectory(ast)
add_subdirectory(Standalone)
add_subdirectory(Test)