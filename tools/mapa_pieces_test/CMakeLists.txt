PROJECT(MAPA_PIECES_Test)

INCLUDE_DIRECTORIES(. ${MAPA_INCLUDE_DIRS})
ADD_DEFINITIONS(-DIGL_HEADER_ONLY)

LINK_DIRECTORIES( ${ANN_LINK_DIR} )

ADD_EXECUTABLE(opts_test opts_test.cpp)
ADD_EXECUTABLE(randsample_test randsample_test.cpp)
ADD_EXECUTABLE(estdim_test estdim_test.cpp)

ADD_EXECUTABLE(mapa_test mapa_test.cpp)
TARGET_LINK_LIBRARIES(mapa_test myann)

ADD_EXECUTABLE(lmsvd_test lmsvd_test.cpp)
TARGET_LINK_LIBRARIES(lmsvd_test myann)

ADD_EXECUTABLE(ann_obj_test ann_obj_test.cpp)
TARGET_LINK_LIBRARIES(ann_obj_test myann)

