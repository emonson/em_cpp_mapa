PROJECT(TOKENIZE_Test)

INCLUDE_DIRECTORIES(. ${MAPA_INCLUDE_DIRS})

if(WIN32)
  INCLUDE_DIRECTORIES( ${MAPA_SOURCE_DIR}/lib/dirent-1.20.1/include/ )
endif(WIN32)

LINK_DIRECTORIES( ${TINYXML2_LINK_DIR} )
LINK_DIRECTORIES( ${ANN_LINK_DIR} )

ADD_EXECUTABLE(ann_tokenize_test ann_tokenize_test.cpp)
TARGET_LINK_LIBRARIES(ann_tokenize_test tinyxml2 myann)

ADD_EXECUTABLE(eigen_tokenize_test eigen_tokenize_test.cpp)
TARGET_LINK_LIBRARIES(eigen_tokenize_test tinyxml2)

ADD_EXECUTABLE(sparse_tokenize_test sparse_tokenize_test.cpp)
TARGET_LINK_LIBRARIES(sparse_tokenize_test tinyxml2)

ADD_EXECUTABLE(dir_tdm_test dir_tdm_test.cpp)

option(BUILD_BOOST_DIR_TOKENIZER "Build Boost dir TOKENIZE test" OFF)
if(BUILD_BOOST_DIR_TOKENIZER)
    FIND_PACKAGE(Boost 1.55 REQUIRED filesystem system)
    ADD_EXECUTABLE(boost_dir_tdm_test boost_dir_tdm_test.cpp)
    TARGET_LINK_LIBRARIES(boost_dir_tdm_test ${Boost_FILESYSTEM_LIBRARIES} ${Boost_SYSTEM_LIBRARIES})
endif(BUILD_BOOST_DIR_TOKENIZER)

ADD_EXECUTABLE(jig_tdm_test jig_tdm_test.cpp)
TARGET_LINK_LIBRARIES(jig_tdm_test tinyxml2)
