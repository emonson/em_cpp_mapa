PROJECT(MAPA_PIECES_Test)

INCLUDE_DIRECTORIES(. ${MAPA_INCLUDE_DIRS})

if(WIN32)
  INCLUDE_DIRECTORIES( ${MAPA_SOURCE_DIR}/lib/dirent-1.20.1/include/ )
endif(WIN32)

ADD_DEFINITIONS(-DIGL_HEADER_ONLY)

LINK_DIRECTORIES( ${ANN_LINK_DIR} ${SVD_LINK_DIR} ${TINYXML2_LINK_DIR} )

ADD_EXECUTABLE(opts_test opts_test.cpp)
ADD_EXECUTABLE(randsample_test randsample_test.cpp)
ADD_EXECUTABLE(estdim_test estdim_test.cpp)
ADD_EXECUTABLE(unique unique.cpp)
ADD_EXECUTABLE(mode mode.cpp)
ADD_EXECUTABLE(logical_indexing logical_indexing.cpp)
ADD_EXECUTABLE(quantile_transfer quantile_transfer.cpp)
ADD_EXECUTABLE(combinations_test combinations_test.cpp)

ADD_EXECUTABLE(mapa_test mapa_test.cpp)
TARGET_LINK_LIBRARIES(mapa_test svd myann)

ADD_EXECUTABLE(mapa_face_clustering_test mapa_face_clustering_test.cpp)
TARGET_LINK_LIBRARIES(mapa_face_clustering_test svd myann)

ADD_EXECUTABLE(mapa_motion_segmentation_test mapa_motion_segmentation_test.cpp)
TARGET_LINK_LIBRARIES(mapa_motion_segmentation_test svd myann)

ADD_EXECUTABLE(lmsvd_test lmsvd_test.cpp)
TARGET_LINK_LIBRARIES(lmsvd_test myann svd)

ADD_EXECUTABLE(ann_obj_test ann_obj_test.cpp)
TARGET_LINK_LIBRARIES(ann_obj_test myann)

ADD_EXECUTABLE(jig_mapa_test jig_mapa_test.cpp)
TARGET_LINK_LIBRARIES(jig_mapa_test tinyxml2 myann svd)

ADD_EXECUTABLE(dir_mapa_test dir_mapa_test.cpp)
TARGET_LINK_LIBRARIES(dir_mapa_test myann svd)

