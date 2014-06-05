PROJECT(DIRENT_Test)

INCLUDE_DIRECTORIES(. ${MAPA_INCLUDE_DIRS})

ADD_EXECUTABLE(ls_dirent ls.c)
ADD_EXECUTABLE(cpp_ls_dirent ls.cpp)
# ADD_EXECUTABLE(find_dirent find.c)
# ADD_EXECUTABLE(locate_dirent locate.c)
# ADD_EXECUTABLE(updatedb_dirent updatedb.c)

