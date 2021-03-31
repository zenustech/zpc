#####
include(CMake/get_cpm.cmake)
#####

# rapidjson
CPMAddPackage(
  NAME rapidjson
  #GIT_TAG f56928de85d56add3ca6ae7cf7f119a42ee1585b
  GIT_TAG 585042c02ba6350e10fc43df8beee1bc097f4c5f
  GITHUB_REPOSITORY Tencent/rapidjson
)
if(rapidjson_ADDED)
  add_library(rapidjson INTERFACE IMPORTED)
  target_include_directories(rapidjson INTERFACE ${rapidjson_SOURCE_DIR}/include)
endif()

# cxxopts
CPMAddPackage(
  NAME cxxopts
  GITHUB_REPOSITORY jarro2783/cxxopts
  VERSION 2.2.0
  OPTIONS
    "CXXOPTS_BUILD_EXAMPLES Off"
    "CXXOPTS_BUILD_TESTS Off"
)

# spdlog
CPMAddPackage(
  NAME spdlog
  VERSION 1.7.0
  GITHUB_REPOSITORY gabime/spdlog
)

# fmt
CPMAddPackage(
  NAME fmt
  GIT_TAG 6.2.1
  GITHUB_REPOSITORY fmtlib/fmt
)

# magic_enum 
CPMAddPackage(
  NAME magic_enum
  GITHUB_REPOSITORY Neargye/magic_enum
  VERSION 0.7.2
)

# catch2
CPMAddPackage(
  NAME Catch2
  GITHUB_REPOSITORY catchorg/Catch2
  VERSION 2.13.4
)
# glm
#CPMAddPackage(
#  NAME glm
#  #VERSION 0.9.9.8
#  GITTAG bf71a834948186f4097caa076cd2663c69a10e1e
#  GITHUB_REPOSITORY g-truc/glm
#)
#if(glm_ADDED) 
#  add_library(glm INTERFACE IMPORTED)
#  target_include_directories(glm INTERFACE ${glm_SOURCE_DIR}/include)
#endif()

# gcem
CPMAddPackage(
  NAME gcem
  VERSION 1.12.0
  #GITTAG 910656e6f09638a8744515f0e7337edcc501e711
  GITHUB_REPOSITORY kthohr/gcem
  DOWNLOAD_ONLY True
)
if(gcem_ADDED) 
  add_library(gcem INTERFACE IMPORTED)
  target_include_directories(gcem INTERFACE ${gcem_SOURCE_DIR}/include)
endif()

# taskflow
CPMAddPackage(
  NAME taskflow
  VERSION 2.7.0
  GITHUB_REPOSITORY taskflow/taskflow
)
if(taskflow_ADDED) 
  add_library(taskflow INTERFACE IMPORTED)
  target_include_directories(taskflow INTERFACE ${taskflow_SOURCE_DIR}/)
endif()
