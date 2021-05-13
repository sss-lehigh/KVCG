add_library(rand SHARED src/zipf.cpp)
target_include_directories(rand PUBLIC include)

add_library(rand_static STATIC src/zipf.cpp)
target_include_directories(rand_static PUBLIC include)