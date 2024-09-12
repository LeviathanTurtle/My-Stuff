cmake_minimum_required(VERSION 3.10)
project(MyStuff)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# helper function to collect all source files recursively
file(GLOB_RECURSE CALCULATORS_SOURCES "Calculators/Source/*.cpp")
file(GLOB_RECURSE GAMES_SOURCES "Games/Source/*.cpp")
file(GLOB_RECURSE TOOLS_SOURCES "Tools/Source/*.cpp")

# add executables for each module
add_executable(CalculatorsExecutable ${CALCULATORS_SOURCES})
add_executable(GamesExecutable ${GAMES_SOURCES})
add_executable(ToolsExecutable ${TOOLS_SOURCES})
