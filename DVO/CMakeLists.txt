# Copyright (c) 2015 Robert Maier
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(DVO_mex)

# RelWithDebInfo, Release or Debug
# SET(CMAKE_BUILD_TYPE RelWithDebInfo FORCE)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

SET(OPENCV_INSTALLED TRUE)
SET(EXEF TRUE)
SET(MEXF TRUE)
SET(CPP_MAIN_FILES src/main.cpp)
SET(CPP_SUBMAIN_FILES src/ImageAlign.cpp)
SET(MEX_MAIN_FILES src/mexMain.cpp)
SET(MEX_SUBMAIN_FILES src/ImageAlignMex.cpp)
SET(SRC_FILES src/dvo.cpp src/dvo.hpp src/tum_benchmark.hpp)
#==============================================================================
# Dependencies
#==============================================================================

# include sophus and Eigen
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/third_party/include/ )

# Eigen
SET(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/third_party/include/)
FIND_PACKAGE(Eigen REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

# OpenCV
#FIND_PACKAGE(OpenCV 2.4.8 COMPONENTS opencv_core opencv_highgui opencv_imgproc REQUIRED)
  #ADD_DEFINITIONS(-DUSE_OPENCV)
  SET(OpenCV_DIR /usr/wiss/sang/.local/share/OpenCV/)
  FIND_PACKAGE( OpenCV REQUIRED)
  #SET(OpenCV_ROOT /usr/share/OpenCV/)
  #FIND_PACKAGE( OpenCV REQUIRED PATHS /usr/share/OpenCV/)
#==============================================================================
# Application
#==============================================================================

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O1 -funroll-loops" )

# preprocessor define: source directory
ADD_DEFINITIONS(-DDVO_SOURCE_DIR=${PROJECT_SOURCE_DIR})


##################FIND MATLAB#########################
######################################################
if(${MEXF})
  #not sure if the following two definitions are necessary
#  ADD_DEFINITIONS(/DMATLAB_MEX_FILE) #define matlab macros
#  ADD_DEFINITIONS(/DMX_COMPAT_32)

  FIND_PACKAGE(Matlab REQUIRED)

  #add definition USE_MEX for C++ code, to be able to use preprocessor if matlab is not used
  ADD_DEFINITIONS(-DUSE_MEX)

  # set up matlab libraries
  INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

  #use correct suffix depending on OS
  if(WIN32) # 32-bit or 64-bit mex
      if (CMAKE_CL_64)
        SET( MEX_SUFFIX .mexw64 )
      else(CMAKE_CL_64)
        SET( MEX_SUFFIX .mexw32 )
      endif(CMAKE_CL_64)
  else(WIN32)
      if (CMAKE_SIZEOF_VOID_P MATCHES "8")
        SET( MEX_SUFFIX .mexa64 )
      else(CMAKE_SIZEOF_VOID_P MATCHES "8")
        SET( MEX_SUFFIX .mexglx )
      endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
  endif(WIN32)
endif(${MEXF})

##################BUILD FILES#########################
######################################################

######################MEX#############################
######################################################
if(${MEXF})
  #if mex lib is build, add suffix MEX to lib file
  SET(MEX_LIB_NAME "${PROJECT_NAME}MEX")
  SET(MEX_SUBLIB_NAME "ImageAlignMEX")

  ADD_LIBRARY(${MEX_LIB_NAME} SHARED ${${MEX_LIB_NAME}} ${MEX_MAIN_FILES} ${SRC_FILES} ${CMAKE_SOURCE_DIR}/Matlabdef.def)
ADD_LIBRARY(${MEX_SUBLIB_NAME} SHARED ${${MEX_SUBLIB_NAME}} ${MEX_SUBMAIN_FILES} ${SRC_FILES} ${CMAKE_SOURCE_DIR}/Matlabdef.def)
  TARGET_LINK_LIBRARIES(${MEX_LIB_NAME}
                        ${MATLAB_LIBRARIES}
                        )
TARGET_LINK_LIBRARIES(${MEX_SUBLIB_NAME}
                        ${MATLAB_LIBRARIES}
                        )
  if (${EIGEN_INSTALLED})
    TARGET_LINK_LIBRARIES(${MEX_LIB_NAME} Eigen3::Eigen) #link to eigen library (available in ${PROJECT_NAME} target (s. below) and ${MEX_LIB_NAME} target )
  endif(${EIGEN_INSTALLED})

    if (${OPENCV_INSTALLED})
    TARGET_LINK_LIBRARIES(${MEX_LIB_NAME} ${OpenCV_LIBS}) #load opencv libs (available only in ${PROJECT_NAME} target)
    TARGET_LINK_LIBRARIES(${MEX_SUBLIB_NAME} ${OpenCV_LIBS})
  endif(${OPENCV_INSTALLED})
 SET_TARGET_PROPERTIES(${MEX_LIB_NAME} PROPERTIES PREFIX "" SUFFIX ${MEX_SUFFIX})
 SET_TARGET_PROPERTIES(${MEX_SUBLIB_NAME} PROPERTIES PREFIX "" SUFFIX ${MEX_SUFFIX})
endif(${MEXF})
##################EXECUTABLE##########################
######################################################
if(${EXEF}) #now take care of the executable
    ADD_EXECUTABLE(${PROJECT_NAME} ${CPP_MAIN_FILES} ${SRC_FILES})
    ADD_EXECUTABLE(ImageAlign ${CPP_SUBMAIN_FILES} ${SRC_FILES})

  if (${OPENCV_INSTALLED})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS}) #load opencv libs (available only in ${PROJECT_NAME} target)
	TARGET_LINK_LIBRARIES(ImageAlign ${OpenCV_LIBS})
  endif(${OPENCV_INSTALLED})

  if (${EIGEN_INSTALLED})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} Eigen3::Eigen) #link to eigen library (available in ${PROJECT_NAME} target and ${MEX_LIB_NAME} target (s. above))
  endif(${EIGEN_INSTALLED})

endif(${EXEF})
