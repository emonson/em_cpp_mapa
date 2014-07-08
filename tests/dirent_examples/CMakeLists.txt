PROJECT(DIRENT_Test)


if(WIN32)
  INCLUDE_DIRECTORIES(. ${MAPA_INCLUDE_DIRS}  ${MAPA_SOURCE_DIR}/lib/dirent-1.20.1/include/ )
else(WIN32)
  INCLUDE_DIRECTORIES(. ${MAPA_INCLUDE_DIRS})
endif(WIN32)

ADD_EXECUTABLE(ls_dirent ls.c)
ADD_EXECUTABLE(cpp_ls_dirent ls.cpp)
# ADD_EXECUTABLE(find_dirent find.c)
# ADD_EXECUTABLE(locate_dirent locate.c)
# ADD_EXECUTABLE(updatedb_dirent updatedb.c)

