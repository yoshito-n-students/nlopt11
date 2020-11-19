project(nlopt11)

include_directories(include)

add_executable(nlopt_tutorial example/nlopt_tutorial.cpp)
target_link_libraries(nlopt_tutorial nlopt)

add_executable(matlab_rosenbrock example/matlab_rosenbrock.cpp)
target_link_libraries(matlab_rosenbrock nlopt)

add_executable(octave_qp5 example/octave_qp5.cpp)
target_link_libraries(octave_qp5 nlopt)