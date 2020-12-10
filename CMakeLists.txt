cmake_minimum_required(VERSION 3.0.2)
project(nlopt11)

if(DEFINED CATKIN_ENV)
  include(CMakeListsROS.txt)
else()
  include(CMakeListsNoROS.txt)
endif()