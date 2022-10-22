# Enable errors for any global constructors.
add_flag_if_supported("-Werror=global-constructors" WERROR_GLOBAL_CONSTRUCTOR)

# our own libs
add_subdirectory(ast)
add_subdirectory(Standalone)
