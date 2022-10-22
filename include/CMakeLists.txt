add_custom_target(Toy)
set_target_properties(Toy PROPERTIES FOLDER Examples) # FOLDER=Examples

include_directories(ast/)
add_subdirectory(Standalone)
