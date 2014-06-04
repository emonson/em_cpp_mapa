PROJECT(TOKENIZE_Test)

INCLUDE_DIRECTORIES(. ${MAPA_INCLUDE_DIRS})
LINK_DIRECTORIES( ${TINYXML2_LINK_DIR} )
LINK_DIRECTORIES( ${ANN_LINK_DIR} )

ADD_EXECUTABLE(ann_tokenize_test ann_tokenize_test.cpp)
TARGET_LINK_LIBRARIES(ann_tokenize_test tinyxml2 myann)

ADD_EXECUTABLE(eigen_tokenize_test eigen_tokenize_test.cpp)
TARGET_LINK_LIBRARIES(eigen_tokenize_test tinyxml2)

ADD_EXECUTABLE(sparse_tokenize_test sparse_tokenize_test.cpp)
TARGET_LINK_LIBRARIES(sparse_tokenize_test tinyxml2)

ADD_EXECUTABLE(tdm_test tdm_test.cpp)
TARGET_LINK_LIBRARIES(tdm_test tinyxml2)
