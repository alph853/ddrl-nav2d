# VTK
# Request only the modules we actually link against to avoid pulling optional
# dependencies (e.g. MPI) via VTK's full module set.
find_package(VTK REQUIRED COMPONENTS
    CommonCore
    CommonDataModel
    FiltersCore
    FiltersGeneral
    FiltersSources
    IOImage
    IOLegacy
    InteractionStyle
    InteractionWidgets
    RenderingAnnotation
    RenderingCore
    RenderingFreeType
    RenderingOpenGL2
    RenderingVolume
    RenderingVolumeOpenGL2
)

add_library(ddrl_visualizer)
add_library(ddrl::visualizer ALIAS ddrl_visualizer)

file(GLOB VTK_WIDGETS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vtk_widgets/*.cpp"
)
file(GLOB VTK_WIDGETS_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/visualizer/widgets/*.hpp"
)
file(GLOB VTK_COMPONENTS_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/visualizer/vtk_components/*.hpp"
)
file(GLOB VTK_COMPONENTS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vtk_components/*.cpp"
)

target_sources(ddrl_visualizer
	PRIVATE
		src/visualizer.cpp
		src/profile_manager.cpp
		src/map_geometry_manager.cpp

		src/provider/event_bus_provider.cpp
		src/provider/mock_provider.cpp
		src/provider/replay_provider.cpp

        ${VTK_WIDGETS_SOURCES}
		${VTK_COMPONENTS_SOURCES}

	PUBLIC
		FILE_SET HEADERS
        BASE_DIRS include
        FILES
			include/visualizer/visualizer.hpp
			include/visualizer/profile_manager.hpp
			include/visualizer/map_geometry_manager.hpp

			include/visualizer/interface/base_provider.hpp

			include/visualizer/provider/event_bus_provider.hpp
			include/visualizer/provider/mock_provider.hpp
			include/visualizer/provider/replay_provider.hpp

            ${VTK_WIDGETS_HEADERS}
			${VTK_COMPONENTS_HEADERS}
)

target_include_directories(ddrl_visualizer
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
)

target_link_libraries(ddrl_visualizer
	PUBLIC
		ddrl::core
		ddrl::logging
		ddrl::sim
		ddrl::events
		ddrl::logging
		# ddrl::rpc
		${VTK_LIBRARIES}
)

target_compile_features(ddrl_visualizer PUBLIC cxx_std_23)

if (DDRL_VIEWER_WITH_QT)
	find_package(Qt6 COMPONENTS Widgets REQUIRED)
	target_link_libraries(ddrl_visualizer PUBLIC Qt6::Widgets)
	target_compile_definitions(ddrl_visualizer PUBLIC DDRL_VIEWER_WITH_QT=1)
endif()

# Enable VTK module autoinit for this target
vtk_module_autoinit(
	TARGETS ddrl_visualizer
	MODULES ${VTK_LIBRARIES}
)
