#Toplevel code for sgerber research code

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

PROJECT(MAPA)

FIND_PACKAGE(BLAS)
FIND_PACKAGE(LAPACK)
FIND_LIBRARY(GFORTRAN_LIBRARY gfortran)

INCLUDE_DIRECTORIES( "/usr/local/include" )

set(CMAKE_CXX_FLAGS "-g")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_OSX_ARCHITECTURES "x86_64")
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9")
set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk")
set(GFORTRAN_LIBRARY "/usr/local/Cellar/gfortran/4.8.2/gfortran/lib/libgfortran.dylib")

set(MAPA_INCLUDE_DIRS 
   ${MAPA_SOURCE_DIR}/lib/ 
   ${MAPA_SOURCE_DIR}/lib/utils 
   ${MAPA_SOURCE_DIR}/lib/statistics 
   ${MAPA_SOURCE_DIR}/lib/metrics  
   ${MAPA_SOURCE_DIR}/lib/linalg
   ${MAPA_SOURCE_DIR}/lib/io 
   ${MAPA_SOURCE_DIR}/lib/tclap 
   ${MAPA_SOURCE_DIR}/lib/ann/
)

set(MAPA_LIB_DIR 
   ${MAPA_SOURCE_DIR}/lib/ 
)

set(ANN_LINK_DIR 
   ${MAPA_SOURCE_DIR}/lib/ann/
)

ADD_SUBDIRECTORY(lib)
ADD_SUBDIRECTORY(tools)

