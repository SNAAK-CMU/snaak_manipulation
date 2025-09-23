cmake_minimum_required(VERSION 3.8)
project(snaak_manipulation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(tf2 REQUIRED) 
find_package(tf2_ros REQUIRED) 
find_package(action_msgs REQUIRED)
# find_package(weight_service REQUIRED)
# find_package(snaak_pneumatic REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ExecuteTrajectory.action"
  "action/Pickup.action"
  "action/ReturnHome.action"
  "action/Place.action"
  DEPENDENCIES geometry_msgs
)

install(PROGRAMS
  scripts/snaak_manipulation_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  ${CMAKE_SOURCE_DIR}/share/${PROJECT_NAME}/
  DESTINATION share/${PROJECT_NAME}/
)

ament_python_install_package(scripts)
ament_package()
