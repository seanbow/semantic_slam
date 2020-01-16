cmake_minimum_required(VERSION 3.0)
project(semantic_slam)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake") 

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 14)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif(CCACHE_FOUND)

# add_compile_options(-g -O3 -march=native -ffast-math -flto=auto -Wall)
add_compile_options(-g -O3 -march=native -ffast-math -Wall -ansi -fdiagnostics-color)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
    roscpp
    roslib
    # gtsam
    cv_bridge
    image_transport
    rosfmt
    tf2_ros
    object_pose_interface_msgs
    darknet_ros_msgs
)

find_package(GTSAM REQUIRED)

find_package(MKL)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package(Boost COMPONENTS filesystem serialization system REQUIRED)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# for google logger and ceres
find_package(gflags REQUIRED)

find_package(OpenCV)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES semantic_slam
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

if (MKL_FOUND)
  include_directories(${MKL_INCLUDE_DIR})
endif()

add_library(SemanticMapper SHARED
  src/Key.cpp
  src/Symbol.cpp
  src/Camera.cpp
  src/CameraSet.cpp
  src/CameraCalibration.cpp
  src/Pose3.cpp
  src/SemanticKeyframe.cpp
  src/Utils.cpp
  src/munkres.cpp
  src/MLDataAssociator.cpp

  src/ceres_cost_terms/ceres_pose_prior.cpp
  src/ceres_cost_terms/ceres_projection.cpp
  src/ceres_cost_terms/ceres_structure.cpp
  src/ceres_cost_terms/ceres_structure_projection.cpp
  src/ceres_cost_terms/ceres_between.cpp
  src/LocalParameterizations.cpp

  src/CeresNode.cpp
  src/CeresFactor.cpp
  src/CeresProjectionFactor.cpp
  src/CeresBetweenFactor.cpp
  src/CeresStructureFactor.cpp
  src/MultiProjectionFactor.cpp
  src/SmartProjectionFactor.cpp
  src/keypoints/gtsam/StructureFactor.cpp
  src/FactorGraph.cpp

  src/keypoints/EstimatedKeypoint.cpp
  src/keypoints/geometry.cpp
  src/keypoints/EstimatedObject.cpp
  src/keypoints/StructureOptimizationProblem.cpp

  src/LoopCloser.cpp
  src/GeometricFeatureHandler.cpp
  src/SemanticSmoother.cpp
  src/SemanticMapper.cpp
  src/ExternalOdometryHandler.cpp

  src/presenters/GeometricMapPresenter.cpp
  src/presenters/SemanticCovisibilityPresenter.cpp
  src/presenters/ObjectMeshPresenter.cpp
  src/presenters/ObjectPosePresenter.cpp
  src/presenters/TrajectoryPresenter.cpp
  src/presenters/PosePresenter.cpp
  src/presenters/ObjectKeypointPresenter.cpp
  src/presenters/GeometricCovisibilityPresenter.cpp
  src/presenters/OdometryTransformPresenter.cpp

  src/feature_tracker/ORBextractor.cpp
  src/feature_tracker/TwoPointRansac.cpp
  src/feature_tracker/FeatureTracker.cpp
  src/feature_tracker/FivePointRansac.cpp
)
target_link_libraries(SemanticMapper
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
  # ${MKL_LIBRARIES}
  gtsam
)


# add_library(UnityMapper unity_mapper.cpp)
# target_link_libraries(UnityMapper
#   ${catkin_LIBRARIES}
#   ${Boost_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   ${CERES_LIBRARIES}
#   # ${MKL_LIBRARIES}
#   gtsam
# )

add_library(SimpleObjectTracker src/SimpleObjectTracker.cpp)
target_link_libraries(SimpleObjectTracker
  ${catkin_LIBRARIES}
)

add_executable(semantic_slam_node src/semantic_slam_node.cpp)
target_link_libraries(semantic_slam_node
SemanticMapper
# UnityMapper
)

add_executable(image_republish_node src/image_republish_node.cpp)
target_link_libraries(image_republish_node
    ${catkin_LIBRARIES}
)

add_executable(kitti_data_node src/kitti_data_node.cpp)
target_link_libraries(kitti_data_node
    ${catkin_LIBRARIES}
)

add_executable(object_tracking_node src/object_tracking_node.cpp)
target_link_libraries(object_tracking_node
  ${catkin_LIBRARIES}
  SimpleObjectTracker
)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############


## Add gtest based cpp test target and link libraries
catkin_add_gtest(testFactorGraph test/testFactorGraph.cpp)
target_link_libraries(testFactorGraph 
    SemanticMapper
    gtsam
    ${CERES_LIBRARIES}
)

catkin_add_gtest(testCeresBasics test/testCeresBasics.cpp)
target_link_libraries(testCeresBasics
    SemanticMapper
    ${CERES_LIBRARIES}
    gtsam
)

catkin_add_gtest(testProjectionFactors test/testProjectionFactors.cpp)
target_link_libraries(testProjectionFactors
    SemanticMapper
    ${CERES_LIBRARIES}
    gtsam
)

catkin_add_gtest(testFactorGraphClone test/testFactorGraphClone.cpp)
target_link_libraries(testFactorGraphClone
    SemanticMapper
    ${CERES_LIBRARIES}
    gtsam
)


# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
