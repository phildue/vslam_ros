

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/ceres_example)

set(sources ${CMAKE_CURRENT_LIST_DIR}/loop_closure_detection.cpp 
${CMAKE_CURRENT_LIST_DIR}/LoopClosureDetection.h
${CMAKE_CURRENT_LIST_DIR}/LoopClosureDetection.cpp)
pd_add_experiment(loop_closure_detection "${sources}")

set(sources2 ${CMAKE_CURRENT_LIST_DIR}/test_pose_graph.cpp)
pd_add_test(pose_graph_optimization "${sources2}")