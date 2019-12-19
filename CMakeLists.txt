cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)

project(tflite-cpp VERSION 1.0.0 LANGUAGES CXX DESCRIPTION "Example code for running Tensorflow Lite on C++")

set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Required because of relocated static libray. We can delete it after we get rid of it.
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -DUSE_SSE4_INSTRUCTIONS=ON")

############## Define Preprocessor Vars ##################
#add_definitions ( -DMODEL_DIR=\"${CMAKE_SOURCE_DIR}/models\")

find_package(Boost 1.58 COMPONENTS program_options system filesystem REQUIRED)
find_package(dlib REQUIRED)

############## Define headers and sourcse ################
set( interfaceHdr
        include/util/utils.h

)

set( interfaceSrc
        src/util/utils.cpp
        #https://github.com/tensorflow/tensorflow/issues/32073 must include workaround, not fixed until recently
        ${CMAKE_SOURCE_DIR}/../tensorflow/tensorflow/lite/tools/evaluation/utils.cc
)

#source_group("Interface Header Files" FILES ${interfaceHdr})
#source_group("Interface Source Files" FILES ${interfaceSrc})

######## linking and properties ########
add_library(${PROJECT_NAME}
        ##### HDR #####
        ${interfaceHdr}

        ##### SRC #####
        ${interfaceSrc}
        )

set_target_properties(${PROJECT_NAME}
        PROPERTIES
            VERSION ${PROJECT_VERSION}
            SOVERSION 1
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
            #CXX_VISIBILITY_PRESET hidden
        #PUBLIC_HEADER
            #include/solver/test.h
        )

target_include_directories(${PROJECT_NAME}
        PUBLIC ${Boost_INCLUDE_DIRS}
        #PUBLIC ${TENSORFLOW_LITE_INCLUDE_DIRS}
        PUBLIC ${CMAKE_SOURCE_DIR}/../tensorflow/
        # Run ./tensorflow/lite/tools/make/download_dependencies.sh in tensorflow dir to d/l flatbuffer deps.
        PUBLIC ${CMAKE_SOURCE_DIR}/../tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include/
        PUBLIC /usr/local/include )

target_link_libraries(${PROJECT_NAME}
        ${Boost_LIBRARIES}
        dlib
        stdc++fs
        ${CMAKE_SOURCE_DIR}/lib/libtensorflowlite.so
        ${CMAKE_SOURCE_DIR}/lib/libtensorflowlite_gpu_gl.so
            #${TESNORFLOW_LITE_LIBRARIES}
        )

###### EXAMPLES ######
add_executable(TestApp examples/test_app.cpp)
target_link_libraries(TestApp ${PROJECT_NAME}
        )

