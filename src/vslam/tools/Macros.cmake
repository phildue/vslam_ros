function(pd_setup_lib name version sources headers namespace)

# Construct library from sources
add_library(${name}
    ${sources}
    ${headers}
    )
set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(${name} PUBLIC cxx_std_17)

# Configure alias so there is no difference whether we link from source/from already built
add_library(${namespace}::${name} ALIAS ${name})

# Set include path (can be different in build/install)
target_include_directories(${name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src/>
    $<INSTALL_INTERFACE:include/vslam/${name}>
    )
target_include_directories(${name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include/>
    $<INSTALL_INTERFACE:include/vslam>
    )

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/
    DESTINATION include/vslam/${name}
    FILES_MATCHING # install only matched files
    PATTERN "*.h" # select header files
    )
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/
    DESTINATION include/vslam/${name}
    FILES_MATCHING # install only matched files
    PATTERN "*.hpp" # select header files
    )  
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION include/vslam/
    FILES_MATCHING # install only matched files
    PATTERN "*.h" # select header files
    )



endfunction()


macro(pd_add_test unit lib)
	add_executable(${unit}Test
			test/test_${unit}.cpp
			)
    target_compile_features(${unit}Test PUBLIC cxx_std_17)

	target_link_libraries(${unit}Test
			PRIVATE
			pd::${lib}
			GTest::gtest_main
			)

	add_test(NAME ${unit}.UnitTest
			COMMAND ${unit}Test
			)

endmacro()

