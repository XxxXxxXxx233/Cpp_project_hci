cmake_minimum_required(VERSION 3.16)
project(hci_project)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_INCLUDE_DIRS C:/OpenCV_MinGW64/build/install/include/opencv2)
FIND_PACKAGE(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(
        ${PROJECT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        #"C:/OpenCV_MinGW64/build/install/include/opencv2"

)

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})