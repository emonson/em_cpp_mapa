PROJECT(MAPA_PIECES_Test)

INCLUDE_DIRECTORIES(. ${MAPA_INCLUDE_DIRS})
ADD_DEFINITIONS(-DIGL_HEADER_ONLY)

LINK_DIRECTORIES( ${ANN_LINK_DIR} )

ADD_EXECUTABLE(opts_test opts_test.cpp)
ADD_EXECUTABLE(randsample_test randsample_test.cpp)
ADD_EXECUTABLE(estdim_test estdim_test.cpp)
ADD_EXECUTABLE(unique unique.cpp)
ADD_EXECUTABLE(mode mode.cpp)
ADD_EXECUTABLE(logical_indexing logical_indexing.cpp)
ADD_EXECUTABLE(quantile_transfer quantile_transfer.cpp)
ADD_EXECUTABLE(combinations_test combinations_test.cpp)

ADD_EXECUTABLE(mapa_test mapa_test.cpp)
TARGET_LINK_LIBRARIES(mapa_test myann)

ADD_EXECUTABLE(mapa_face_clustering_test mapa_face_clustering_test.cpp)
TARGET_LINK_LIBRARIES(mapa_face_clustering_test myann)

ADD_EXECUTABLE(mapa_motion_segmentation_test mapa_motion_segmentation_test.cpp)
TARGET_LINK_LIBRARIES(mapa_motion_segmentation_test myann)

ADD_EXECUTABLE(lmsvd_test lmsvd_test.cpp)
TARGET_LINK_LIBRARIES(lmsvd_test myann)

ADD_EXECUTABLE(ann_obj_test ann_obj_test.cpp)
TARGET_LINK_LIBRARIES(ann_obj_test myann)

