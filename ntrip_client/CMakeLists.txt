cmake_minimum_required(VERSION 2.8.3)
project(ntrip_client)

find_package( catkin REQUIRED COMPONENTS rospy)


### execute the setup script, so that this package builds via catkin
catkin_python_setup()


catkin_package(
  CATKIN_DEPENDS
)

install(DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  USE_SOURCE_PERMISSIONS)

install(DIRECTORY launch/ 
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
