cmake_minimum_required(VERSION 3.8)
project(fabric_cbf_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(fabric_obstacle_distance REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CbfDebug.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

install(PROGRAMS
  scripts/fabric_cbf_controller_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(
  rclpy
  std_msgs
  geometry_msgs
  sensor_msgs
  fabric_obstacle_distance
  rosidl_default_runtime
)

ament_package()
