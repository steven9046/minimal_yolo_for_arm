cmake_minimum_required(VERSION 3.0)

# set(CMAKE_SYSTEM_PROCESSOR "x86_64")
set(CMAKE_SYSTEM_PROCESSOR "aarch64")
# add_definitions(-DARMNN_DELEGATE)

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
	set (SUB_LIB_DIR "x64_linux")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
	set (SUB_LIB_DIR "aarch64")
endif()

# Compile option
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)

# Create project
set(ProjectName "ssd")
project(${ProjectName})

# For OpenCV
find_package(OpenCV REQUIRED)

# Create executable file
add_executable(${ProjectName} main.cpp)
# Create InferenceHelper library both armnn and tflite
set(TENSORFLOW_LIB ${CMAKE_CURRENT_LIST_DIR}/tensorflow_prebuilt/lib/${SUB_LIB_DIR}/libtensorflowlite.so)

# main
target_include_directories(${ProjectName} PUBLIC
                           ${OpenCV_INCLUDE_DIRS}
                           ${CMAKE_CURRENT_LIST_DIR}/tensorflow_prebuilt/tensorflow
                           ${CMAKE_CURRENT_LIST_DIR}/tensorflow_prebuilt/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
                           ${CMAKE_CURRENT_LIST_DIR}/tensorflow_prebuilt/tensorflow/tensorflow/lite/tools/make/downloads/absl
                        #    ${CMAKE_CURRENT_LIST_DIR}/armnn_prebuilt/include
                        #    ${CMAKE_CURRENT_LIST_DIR}/armnn_delegate_prebuilt/include
                           )

target_link_libraries(${ProjectName} 
                        ${TENSORFLOW_LIB} 
                        ${OpenCV_LIBS}
                        # ${CMAKE_CURRENT_LIST_DIR}/armnn_prebuilt/lib/${SUB_LIB_DIR}/libarmnn.so
                        # ${CMAKE_CURRENT_LIST_DIR}/armnn_delegate_prebuilt/lib/${SUB_LIB_DIR}/libarmnnDelegate.so
                        )

file(COPY ${CMAKE_CURRENT_LIST_DIR}/resource DESTINATION ${CMAKE_BINARY_DIR}/)
add_definitions(-DRESOURCE_DIR="${CMAKE_BINARY_DIR}/resource/")
