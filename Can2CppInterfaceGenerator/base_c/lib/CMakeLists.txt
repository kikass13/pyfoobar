cmake_minimum_required(VERSION 2.8.3)
project(<>AUTO_INSERT:PACKAGE_NAME<>)

# Set to Release in order to speed up the program significantly
if(NOT CMAKE_BUILD_TYPE)
  #set(CMAKE_BUILD_TYPE Release) #None, Debug, Release, RelWithDebInfo, MinSizeRel
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ executable
add_library(${PROJECT_NAME} src/can_interface.cpp src/can_encode_decode.cpp ${HEADER_FILES})
#add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
#target_link_libraries(${PROJECT_NAME}
#)

#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS ${PROJECT_NAME}
)
