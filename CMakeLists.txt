﻿# CMakeList.txt: проект CMake для parallelmatrix; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)

# Включение горячей перезагрузки для компиляторов MSVC, если поддерживается.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("parallelmatrix")

# Добавьте источник в исполняемый файл этого проекта.
add_executable (parallelmatrix1 "lab1/parallelmatrix.cpp" "lab1/parallelmatrix.h")
add_executable (parallelmatrix2 "lab2/parallelmatrix2.cpp" "lab2/parallelmatrix.h")
add_executable (parallelmatrix4 "lab4/cudamatrix.cu")


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET parallelmatrix1 PROPERTY CXX_STANDARD 20)
  set_property(TARGET parallelmatrix2 PROPERTY CXX_STANDARD 20)
endif()

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
