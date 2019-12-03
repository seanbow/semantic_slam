cmake_minimum_required(VERSION 3.0)
project(semantic_slam)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 14)

add_compile_options(-g3 -O3 -march=native)
# add_compile_options(-g3 -O3)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
    roscpp
    roslib
    gtsam
    cv_bridge
    image_transport
    rosfmt
    object_pose_interface_msgs
    darknet_ros_msgs
)

find_package(GTSAM REQUIRED)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package(Boost COMPONENTS filesystem serialization system REQUIRED)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

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
)

## Declare a C++ library
add_library(FactorGraph
  src/FactorGraph.cpp
)
target_link_libraries(FactorGraph
    Eigen3::Eigen
    ${catkin_LIBRARIES}
    ${Boost_LIBRARIES}
    # gtsam
)

add_library(Utils src/Utils.cpp)

add_library(Key src/Key.cpp)
add_library(Symbol src/Symbol.cpp)
add_library(CameraCalibration src/CameraCalibration.cpp)
add_library(Camera src/Camera.cpp)
add_library(CameraSet src/CameraSet.cpp)

## Feature Tracker ##
add_library(TwoPointRansac src/feature_tracker/TwoPointRansac.cpp)
add_library(FivePointRansac src/feature_tracker/FivePointRansac.cpp)
add_library(ORBextractor src/feature_tracker/ORBextractor.cpp)
add_library(FeatureTracker src/feature_tracker/FeatureTracker.cpp)
target_link_libraries(FeatureTracker
  ORBextractor
  FivePointRansac
  TwoPointRansac
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_library(StructureFactor src/keypoints/gtsam/StructureFactor.cpp)
# add_library(StructureProjectionFactor src/keypoints/gtsam/StructureProjectionFactor.cpp)

# add_executable(test_structure_factor src/test_structure_factor.cpp)
# target_link_libraries(test_structure_factor
#     ${catkin_LIBRARIES}
#     StructureFactor
#     Utils
#     # gtsam
# )

set(BASE_LIBS 
    Utils
    Key
    Symbol
    CameraCalibration
)

set(CERES_FACTORS
  CeresProjectionFactor
  CeresBetweenFactor
  CeresStructureFactor
)

add_library(CeresProjectionFactor src/CeresProjectionFactor.cpp)
target_link_libraries(CeresProjectionFactor
  ${CERES_LIBRARIES}
)

add_library(CeresBetweenFactor src/CeresBetweenFactor.cpp)
target_link_libraries(CeresBetweenFactor
  ${CERES_LIBRARIES}
)

add_library(CeresStructureFactor src/CeresStructureFactor.cpp)
target_link_libraries(CeresStructureFactor
  StructureFactor
  ${CERES_LIBRARIES}
)

add_library(ExternalOdometryHandler src/ExternalOdometryHandler.cpp)
target_link_libraries(ExternalOdometryHandler
  CeresBetweenFactor
  ${CERES_LIBRARIES}
)

add_library(MultiProjectionFactor src/MultiProjectionFactor.cpp)
target_link_libraries(MultiProjectionFactor
  CameraSet
  ${CERES_LIBRARIES}
  ${BASE_LIBS}
)

add_library(SmartProjectionFactor src/SmartProjectionFactor.cpp)
target_link_libraries(SmartProjectionFactor
  ${CERES_LIBRARIES}
  ${BASE_LIBS}
)

add_library(GeometricFeatureHandler src/GeometricFeatureHandler.cpp)
target_link_libraries(GeometricFeatureHandler
  FeatureTracker
  MultiProjectionFactor
  SmartProjectionFactor
)

add_library(PosePresenter src/PosePresenter.cpp)
add_library(TrajectoryPresenter src/TrajectoryPresenter.cpp)
add_library(ObjectMeshPresenter src/ObjectMeshPresenter.cpp)
add_library(GeometricMapPresenter src/GeometricMapPresenter.cpp)

add_library(EstimatedObject src/keypoints/EstimatedObject.cpp)
target_link_libraries(EstimatedObject
  Camera
)

add_library(EstimatedKeypoint src/keypoints/EstimatedKeypoint.cpp)
#target_link_libraries(EstimatedKeypoint
#    ${BASE_LIBS}
 #   ${CERES_FACTORS}
#)

add_library(SemanticKeyframe src/SemanticKeyframe.cpp)

add_library(munkres src/munkres.cpp)

add_library(SimpleObjectTracker src/SimpleObjectTracker.cpp)
target_link_libraries(SimpleObjectTracker
  ${catkin_LIBRARIES}
)

add_library(MLDataAssociator src/MLDataAssociator.cpp)
target_link_libraries(MLDataAssociator
  munkres
)

add_library(geometry src/keypoints/geometry.cpp)
# target_link_libraries(geometry 
#   gtsam
# )

add_library(SemanticMapper src/SemanticMapper.cpp)
target_link_libraries(SemanticMapper
  geometry
  EstimatedObject
  EstimatedKeypoint
  StructureOptimizationProblem
  Utils
  GeometricFeatureHandler
  MLDataAssociator
#  SemanticKeyframe
  FactorGraph
  ${CERES_FACTORS}
#  ${CERES_LIBRARIES}
#  ${BASE_LIBS}
)


add_library(StructureOptimizationProblem src/keypoints/StructureOptimizationProblem.cpp)


## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(test_geometry src/test_geometry.cpp)
# target_link_libraries(test_geometry
#     ${catkin_LIBRARIES}
#     Utils
#     # gtsam
#     geometry
#     ${CERES_LIBRARIES}
#     ${BASE_LIBS}
# )

add_executable(semantic_slam_node src/semantic_slam_node.cpp)
target_link_libraries(semantic_slam_node
    ${catkin_LIBRARIES}
    # ${GTSAM_LIBRARIES}
    ${Boost_LIBRARIES}
    ${CERES_LIBRARIES}
    StructureOptimizationProblem
    Camera
    MLDataAssociator
    FactorGraph
    CameraSet
    geometry
    munkres
    FactorGraph
    ExternalOdometryHandler
    GeometricFeatureHandler
    FeatureTracker
    SemanticMapper
    PosePresenter
    TrajectoryPresenter
    ObjectMeshPresenter
    GeometricMapPresenter
    SemanticKeyframe
    gtsam
    ${BASE_LIBS}
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
    FactorGraph
    ${CERES_LIBRARIES}
    ${BASE_LIBS}
)

catkin_add_gtest(testCeresBasics test/testCeresBasics.cpp)
target_link_libraries(testCeresBasics
    FactorGraph
    CeresBetweenFactor
    ${CERES_LIBRARIES}
    # gtsam
    ${BASE_LIBS}
)

catkin_add_gtest(testProjectionFactors test/testProjectionFactors.cpp)
target_link_libraries(testProjectionFactors
    FactorGraph
    MultiProjectionFactor
    SmartProjectionFactor
    CeresProjectionFactor
    ${CERES_LIBRARIES}
    ${BASE_LIBS}
)


# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
