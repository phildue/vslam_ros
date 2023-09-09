

set(sources
    ${CMAKE_CURRENT_LIST_DIR}/AlignmentPoseGraph.cpp
    ${CMAKE_CURRENT_LIST_DIR}/overlays.cpp
    )

pd_add_experiment(rgbd_odometry_local_map ${CMAKE_CURRENT_LIST_DIR}/rgbd_odometry_local_map.cpp ${sources})
pd_add_experiment(multiframe_alignment ${CMAKE_CURRENT_LIST_DIR}/rgbd_odometry_local_map.cpp ${sources})
