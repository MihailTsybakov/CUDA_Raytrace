cmake_minimum_required(VERSION 3.1)

project(Task_5_Sem_4)

include_directories(${PROJECT_SOURCE_DIR})

set(TASK_5_SRC raytrace.cu)
set(TASK_5_HDR canvas.h)

add_executable(${PROJECT_NAME} ${TASK_5_SRC} ${TASK_5_HDR})