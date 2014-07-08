PROJECT(SVDLIBC_Test)

INCLUDE_DIRECTORIES(. ${MAPA_INCLUDE_DIRS})
ADD_DEFINITIONS(-DIGL_HEADER_ONLY)

LINK_DIRECTORIES( ${SVD_LINK_DIR} ${REDSVD_LINK_DIR} )

ADD_EXECUTABLE(eigen_sparse_fill eigen_sparse_fill.cpp)
TARGET_LINK_LIBRARIES(eigen_sparse_fill svd)

ADD_EXECUTABLE(sparse_svd sparse_svd.cpp)
TARGET_LINK_LIBRARIES(sparse_svd svd)

ADD_EXECUTABLE(dense_svd dense_svd.cpp)
TARGET_LINK_LIBRARIES(dense_svd svd)

ADD_EXECUTABLE(svdmain main.c)
TARGET_LINK_LIBRARIES(svdmain svd)

ADD_EXECUTABLE(jig_svd_test jig_svd_test.cpp)
TARGET_LINK_LIBRARIES(jig_svd_test tinyxml2 svd)

# ADD_EXECUTABLE(jig_red_test jig_red_test.cpp)
# TARGET_LINK_LIBRARIES(jig_red_test tinyxml2 svd redsvd)

ADD_EXECUTABLE(dir_svd_test dir_svd_test.cpp)
TARGET_LINK_LIBRARIES(dir_svd_test svd)
