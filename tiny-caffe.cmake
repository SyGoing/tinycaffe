# mini-caffe.cmake

option(USE_CUDA "Use CUDA support" ON)
option(USE_CUDNN "Use CUDNN support" ON)
option(USE_JAVA "Use JAVA support" OFF)

# select BLAS
set(BLAS "openblas" CACHE STRING "Selected BLAS library")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/Cuda.cmake)

if(USE_JAVA)
  find_package(JNI)
endif()

# turn on C++11
if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# include and library
if(MSVC)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/include
                      ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include/opencv/include)
  link_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/libs/release)
  list(APPEND Caffe_LINKER_LIBS  libprotobuf glog gflags
                                libopenblas opencv_world310)
elseif(ANDROID)
  # TODO https://github.com/android-ndk/ndk/issues/105
  set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -nodefaultlibs -lgcc -lc -lm -ldl")
  if(ANDROID_EXTRA_LIBRARY_PATH)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/include
                        ${ANDROID_EXTRA_LIBRARY_PATH}/include)
    link_directories(${ANDROID_EXTRA_LIBRARY_PATH}/lib)
    list(APPEND Caffe_LINKER_LIBS openblas protobuf log)
  else(ANDROID_EXTRA_LIBRARY_PATH)
    message(FATAL_ERROR "ANDROID_EXTRA_LIBRARY_PATH must be set.")
  endif(ANDROID_EXTRA_LIBRARY_PATH)
else(MSVC)
  if(APPLE)
    include_directories(/usr/local/opt/openblas/include)
    link_directories(/usr/local/opt/openblas/lib)
  endif(APPLE)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
  list(APPEND Caffe_LINKER_LIBS protobuf)
  if(BLAS STREQUAL "openblas")
    list(APPEND Caffe_LINKER_LIBS openblas)
    message(STATUS "Use OpenBLAS for blas library")
  else()
    list(APPEND Caffe_LINKER_LIBS blas)
    message(STATUS "Use BLAS for blas library")
  endif()
endif(MSVC)

# source file structure
if(HAVE_CUDA)
	file(GLOB CAFFE_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp
						${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
						${CMAKE_CURRENT_LIST_DIR}/src/caffe.pb.h
						${CMAKE_CURRENT_LIST_DIR}/src/caffe.pb.cc
						${CMAKE_CURRENT_LIST_DIR}/src/*.cuh)
else()
	file(GLOB CAFFE_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp
						${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
						${CMAKE_CURRENT_LIST_DIR}/src/caffe.pb.h
						${CMAKE_CURRENT_LIST_DIR}/src/caffe.pb.cc)
endif()
# cpp code
set(CAFFE_COMPILE_CODE ${CAFFE_SRC})

# cuda support
if(HAVE_CUDA)
  message(STATUS "We have CUDA support")
  # cuda code
  file(GLOB CAFFE_SRC_ALL_CU ${CMAKE_CURRENT_LIST_DIR}/src/*.cu)
  set(CAFFE_CUDA_CODE ${CAFFE_SRC_ALL_CU})
  # cudnn support
  if(HAVE_CUDNN)
    message(STATUS "We have CUDNN support")
    # source file structure
    file(GLOB CAFFE_SRC_LAYERS_CUDNN ${CMAKE_CURRENT_LIST_DIR}/src/cudnn/*.hpp
                                     ${CMAKE_CURRENT_LIST_DIR}/src/cudnn/*.cpp
                                     ${CMAKE_CURRENT_LIST_DIR}/src/cudnn/*.cu)
    # cuda code
    file(GLOB CAFFE_CUDNN_CUDA_CODE ${CMAKE_CURRENT_LIST_DIR}/src/cudnn/*.cu)
    list(APPEND CAFFE_CUDA_CODE ${CAFFE_CUDNN_CUDA_CODE})
    # cpp code
    file(GLOB CAFFR_CUDNN_CPP_CODE ${CMAKE_CURRENT_LIST_DIR}/src/cudnn/*.hpp
                                   ${CMAKE_CURRENT_LIST_DIR}/src/cudnn/*.cpp)
    list(APPEND CAFFE_COMPILE_CODE ${CAFFR_CUDNN_CPP_CODE})
    set(CAFFE_SRC_LAYERS_CUDNN ${CAFFR_CUDNN_CPP_CODE}
                               ${CAFFE_CUDNN_CUDA_CODE})
  endif()
  caffe_cuda_compile(CAFFE_CUDA_OBJS ${CAFFE_CUDA_CODE})
  list(APPEND CAFFE_COMPILE_CODE ${CAFFE_CUDA_OBJS})
endif()

# java support
if(JNI_FOUND)
  message(STATUS "We have JAVA support")
  file(GLOB CAFFE_SRC_JNI ${CMAKE_CURRENT_LIST_DIR}/src/jni/*.h
                          ${CMAKE_CURRENT_LIST_DIR}/src/jni/*.c)
  list(APPEND CAFFE_COMPILE_CODE ${CAFFE_SRC_JNI})
  include_directories(${JNI_INCLUDE_DIRS})
  list(APPEND Caffe_LINKER_LIBS ${JNI_LIBRARIES})
endif()

# android jni
if(ANDROID)
  message(STATUS "We have JAVA support")
  file(GLOB CAFFE_SRC_JNI ${CMAKE_CURRENT_LIST_DIR}/src/jni/*.h
                          ${CMAKE_CURRENT_LIST_DIR}/src/jni/*.c)
  list(APPEND CAFFE_COMPILE_CODE ${CAFFE_SRC_JNI})
endif()

# file structure
source_group(src FILES ${CAFFE_SRC})
source_group(src\\cudnn FILES ${CAFFE_SRC_LAYERS_CUDNN})

add_definitions(-DCAFFE_EXPORTS)
add_definitions(-DUSE_OPENCV)
add_library(caffe SHARED ${CAFFE_COMPILE_CODE})
target_link_libraries(caffe ${Caffe_LINKER_LIBS})
