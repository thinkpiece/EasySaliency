include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
)

if( BUILD_DEMO )
    add_executable(saliency_demo demo.cc)
    target_link_libraries(saliency_demo easysaliency_p ${OpenCV_LIBS})
endif( BUILD_DEMO )
