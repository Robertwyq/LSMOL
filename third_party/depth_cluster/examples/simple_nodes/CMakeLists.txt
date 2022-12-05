find_package(pybind11 REQUIRED)
pybind11_add_module(segment segment.cpp ndarray_converter.cpp)
target_link_libraries(segment PRIVATE
        image_labeler
        velodyne_utils
        projections
        ground_remove
        folder_reader
        ${Boost_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        ${CMAKE_THREAD_LIBS_INIT})