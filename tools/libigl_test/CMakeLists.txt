PROJECT(LIBIGL_Test)

INCLUDE_DIRECTORIES(. ${MAPA_INCLUDE_DIRS})
ADD_DEFINITIONS(-DIGL_HEADER_ONLY)

ADD_EXECUTABLE(cat cat.cpp)
ADD_EXECUTABLE(dmat dmat.cpp)
ADD_EXECUTABLE(hello hello.cpp)
ADD_EXECUTABLE(norm norm.cpp)
ADD_EXECUTABLE(reshape reshape.cpp)
ADD_EXECUTABLE(slice slice.cpp)
ADD_EXECUTABLE(sort sort.cpp)
ADD_EXECUTABLE(sortrows sortrows.cpp)

