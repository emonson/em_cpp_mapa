PROJECT(MAPA_PIECES_Test)

INCLUDE_DIRECTORIES(. ${MAPA_INCLUDE_DIRS})
ADD_DEFINITIONS(-DIGL_HEADER_ONLY)

LINK_DIRECTORIES( ${ANN_LINK_DIR} )

ADD_EXECUTABLE(opts_test opts_test.cpp)
ADD_EXECUTABLE(randsample_test randsample_test.cpp)

ADD_EXECUTABLE(ann_test ann_test.cpp)
TARGET_LINK_LIBRARIES(ann_test myann)

ADD_EXECUTABLE(ann_obj_test ann_obj_test.cpp)
TARGET_LINK_LIBRARIES(ann_obj_test myann)

