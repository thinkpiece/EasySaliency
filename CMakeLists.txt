# Junyoung Park <junyoung.park@kaist.ac.kr>
# sep-2013: initial version
# feb-2014: modified to light version

cmake_minimum_required( VERSION 2.8 )

project( easysaliency )

# library version control
set( PROJECT_VERSION_MAJOR "0")
set( PROJECT_VERSION_MINOR "1")
set( PROJECT_SOVERSION ${PROJECT_VERSION_MAJOR} )
set( PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR} )

# library configuration file
configure_file (
  "${PROJECT_SOURCE_DIR}/esconfig.h.in"
  "${PROJECT_SOURCE_DIR}/esconfig.h"
)

# include additional path to find header files
include_directories("${PROJECT_SOURCE_DIR}")

# check the required libraries
find_package(OpenCV REQUIRED)

# shared library install option
set( LIB_SUFFIX "" CACHE STRING "Define suffix of directory name (32/64)" )
set( EXEC_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE PATH "Installation prefix for executables and object code libraries" FORCE )
set( BIN_INSTALL_DIR ${EXEC_INSTALL_PREFIX}/bin CACHE PATH "Installation prefix for user executables" FORCE )
set( LIB_INSTALL_DIR ${EXEC_INSTALL_PREFIX}/lib${LIB_SUFFIX} CACHE PATH  "Installation prefix for object code libraries" FORCE )
set( INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include CACHE PATH "Installation prefix for C header files" FORCE )

option( BUILD_SHARED_LIBS "Build shared libraries" OFF )
option( BUILD_DEMO "Build demo" ON )

if( CMAKE_COMPILER_IS_GNUCXX )
  if( CMAKE_BUILD_TYPE STREQUAL "Release" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2" )
  endif()

  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall" )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall" )
endif()
if ( CMAKE_COMPILER_IS_GNUCXX AND BUILD_SHARED_LIBS )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" )
endif()

# libraries directory
add_subdirectory( src )

# demo directory
if( BUILD_DEMO )
  add_subdirectory( demo )
endif( BUILD_DEMO )

# summary
message(STATUS "")
message(STATUS "General configuration for EasySaliency v${PROJECT_VERSION}")
message(STATUS "")
message(STATUS "  Platform:")
message(STATUS "    Host:             " ${CMAKE_HOST_SYSTEM_NAME}
                ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR})

if(CMAKE_CROSSCOMPILING)
  message(STATUS "    Target:         " ${CMAKE_SYSTEM_NAME}
                  ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
endif()

message(STATUS "    CMake:            " ${CMAKE_VERSION})
message(STATUS "    CMake generator:  " ${CMAKE_GENERATOR})
message(STATUS "    CMake build tool: " ${CMAKE_BUILD_TOOL})
if(MSVC)
  message(STATUS "    MSVC:           " ${MSVC_VERSION})
endif()
if(CMAKE_GENERATOR MATCHES Xcode)
  message(STATUS "    Xcode:          " ${XCODE_VERSION})
endif()
if(NOT CMAKE_GENERATOR MATCHES "Xcode|Visual Studio")
  message(STATUS "    Configuration:  " ${CMAKE_BUILD_TYPE})
endif()

# ========================== C/C++ options ==========================
message(STATUS "")
message(STATUS "  C/C++:")
message(STATUS "    C++ Compiler:           " ${OPENCV_COMPILER_STR})
message(STATUS "    C++ flags (Release):    " ${CMAKE_CXX_FLAGS}
                                              ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "    C++ flags (Debug):      " ${CMAKE_CXX_FLAGS}
                                              ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "    C Compiler:             " ${CMAKE_C_COMPILER}
                                              ${CMAKE_C_COMPILER_ARG1})
message(STATUS "    C flags (Release):      " ${CMAKE_C_FLAGS}
                                              ${CMAKE_C_FLAGS_RELEASE})
message(STATUS "    C flags (Debug):        " ${CMAKE_C_FLAGS}
                                              ${CMAKE_C_FLAGS_DEBUG})
if(WIN32)
  message(STATUS "    Linker flags (Release): " ${CMAKE_EXE_LINKER_FLAGS}
                                                ${CMAKE_EXE_LINKER_FLAGS_RELEASE})
  message(STATUS "    Linker flags (Debug):   " ${CMAKE_EXE_LINKER_FLAGS}
                                                ${CMAKE_EXE_LINKER_FLAGS_DEBUG})
else()
  message(STATUS "    Linker flags (Release): " ${CMAKE_SHARED_LINKER_FLAGS}
                                                ${CMAKE_SHARED_LINKER_FLAGS_RELEASE})
  message(STATUS "    Linker flags (Debug):   " ${CMAKE_SHARED_LINKER_FLAGS}
                                                ${CMAKE_SHARED_LINKER_FLAGS_DEBUG})
endif()

message(STATUS "")
message(STATUS "  EasySaliency shared library path (CMAKE_INSTALL_PREFIX):")
message(STATUS "    Path for user executables: "      ${BIN_INSTALL_DIR})
message(STATUS "    Path for object code libraries: " ${LIB_INSTALL_DIR})
message(STATUS "    Path for C header files: "        ${INCLUDE_INSTALL_DIR})
message(STATUS "")

message(STATUS "")
message(STATUS "  EasySaliency build option:")
message(STATUS "    Shared library (BUILD_SHARED_LIBS): "  ${BUILD_SHARED_LIBS})
message(STATUS "    Demo (BUILD_DEMO): "                   ${BUILD_DEMO})
message(STATUS "")
