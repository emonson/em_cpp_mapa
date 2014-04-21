#Toplevel code for sgerber research code

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)


IF(DEFINED CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of
  build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug
  Release RelWithDebInfo MinSizeRel.")
ELSE()
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build,
  options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release
  RelWithDebInfo MinSizeRel.")
ENDIF()
set(CMAKE_OSX_ARCHITECTURES x86_64 CACHE STRING "")
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.9 CACHE STRING "")
set(CMAKE_OSX_SYSROOT /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk CACHE STRING "")
set(GFORTRAN_LIBRARY /usr/local/Cellar/gfortran/4.7.2/gfortran/lib/libgfortran.dylib CACHE STRING "")

PROJECT(MAPA)

FIND_PACKAGE(BLAS)
FIND_PACKAGE(LAPACK)
FIND_LIBRARY(GFORTRAN_LIBRARY gfortran)

INCLUDE_DIRECTORIES( "/usr/local/include" )

set(MAPA_INCLUDE_DIRS 
   ${MAPA_SOURCE_DIR}/lib/ 
   ${MAPA_SOURCE_DIR}/lib/utils 
   ${MAPA_SOURCE_DIR}/lib/statistics 
   ${MAPA_SOURCE_DIR}/lib/metrics  
   ${MAPA_SOURCE_DIR}/lib/linalg
   ${MAPA_SOURCE_DIR}/lib/igl 
   ${MAPA_SOURCE_DIR}/lib/io 
   ${MAPA_SOURCE_DIR}/lib/tclap 
   ${MAPA_SOURCE_DIR}/lib/ann/
   ${MAPA_SOURCE_DIR}/lib/ann/ANN/
   ${MAPA_SOURCE_DIR}/lib/tinyxml2/
   ${MAPA_SOURCE_DIR}/lib/boost_1_55_0/
   ${MAPA_SOURCE_DIR}/lib/kmeans/
)

set(MAPA_LIB_DIR 
   ${MAPA_SOURCE_DIR}/lib/ 
)

set(ANN_LINK_DIR 
   ${MAPA_SOURCE_DIR}/lib/ann/
)
set(TINYXML2_LINK_DIR 
   ${MAPA_SOURCE_DIR}/lib/tinyxml2/
)

ADD_SUBDIRECTORY(lib)
ADD_SUBDIRECTORY(tools)

