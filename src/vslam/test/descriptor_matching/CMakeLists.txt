

set(sources 
    ${CMAKE_CURRENT_LIST_DIR}/BundleAdjustment.cpp
    ${CMAKE_CURRENT_LIST_DIR}/overlays.cpp
    ${CMAKE_CURRENT_LIST_DIR}/FeatureTracking.cpp
    ${CMAKE_CURRENT_LIST_DIR}/FeatureTrackingOcv.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Matcher.cpp)

pd_add_test(ceres_ba ${CMAKE_CURRENT_LIST_DIR}/bundle_adjustment/test_ceres_ba.cpp)
pd_add_test(ba ${CMAKE_CURRENT_LIST_DIR}/bundle_adjustment/test_ba.cpp ${CMAKE_CURRENT_LIST_DIR}/BundleAdjustment.cpp)

pd_add_experiment(matching ${CMAKE_CURRENT_LIST_DIR}/test_matching.cpp "${sources}")
pd_add_experiment(descriptor_odometry ${CMAKE_CURRENT_LIST_DIR}/test_odometry.cpp "${sources}")
pd_add_experiment(match_and_optimize ${CMAKE_CURRENT_LIST_DIR}/test_match_and_optimize.cpp "${sources}")
