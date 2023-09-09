
pd_add_test(jacobians ${CMAKE_CURRENT_LIST_DIR}/test_jacobians.cpp)
pd_add_experiment(rgbd_odometry_simple ${CMAKE_CURRENT_LIST_DIR}/rgbd_odometry_simple.cpp)
pd_add_experiment(rgb_odometry_simple ${CMAKE_CURRENT_LIST_DIR}/rgb_odometry_simple.cpp)
pd_add_experiment(rgbd_odometry_entropy ${CMAKE_CURRENT_LIST_DIR}/rgbd_odometry_entropy.cpp)
pd_add_experiment(direct_icp ${CMAKE_CURRENT_LIST_DIR}/test_direct_icp.cpp)
