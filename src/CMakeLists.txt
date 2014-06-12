# Library CMake file
include_directories( ${PROJECT_SOURCE_DIR} )

# soiurce files
set( easysaliency_SRCS
     normalizer.cc
     scalespace.cc
     saliency.cc
     imgdump.cc
)

# static library

add_library( easysaliency_p STATIC ${easysaliency_SRCS} )
target_link_libraries( easysaliency_p ${OpenCV_LIBS} )

# shared library
# set public header
set( easysaliency_HDRS
     normalizer.h
)

add_library( easysaliency SHARED ${easysaliency_SRCS} ${easysaliency_HDRS} )
set_target_properties( easysaliency PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_SOVERSION}
  INSTALL_NAME_DIR ${LIB_INSTALL_DIR}
  DEFINE_SYMBOL CV_EXPORTS
)

target_link_libraries( easysaliency ${OpenCV_LIBS} )

install( TARGETS easysaliency
  FRAMEWORK DESTINATION ${FRAMEWORK_INSTALL_DIR}
  LIBRARY DESTINATION ${LIB_INSTALL_DIR}
  RUNTIME DESTINATION ${BIN_INSTALL_DIR}
  ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
)

install( FILES ${easysaliency_HDRS} DESTINATION ${INCLUDE_INSTALL_DIR}/easysaliency )
