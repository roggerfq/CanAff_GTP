cmake_minimum_required(VERSION 2.8)  

project(TestProject)    

find_package( OpenCV REQUIRED )


#Esto es requerido para poder usar openMP
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DFOO)
# create an executable and a library target, both requiring automoc:


set(CMAKE_BUILD_TYPE Release)
add_executable(main main.cpp)
target_link_libraries(main -fopenmp ${OpenCV_LIBS})
