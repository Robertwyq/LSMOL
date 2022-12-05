#include <cstdio>

#include <string>
#include <thread>

#include <pybind11/pybind11.h>
#include "ndarray_converter.h"

#include "clusterers/image_based_clusterer.h"

#include "ground_removal/depth_ground_remover.h"
#include "projections/projection_params.h"
#include "utils/cloud.h"
#include "utils/radians.h"
#include "utils/timer.h"


using std::string;
using std::to_string;

using namespace depth_clustering;


class Segment {
    using GroundRemover = DepthGroundRemover;
    using GroundRemoverPtr = std::shared_ptr<GroundRemover>;
    using Clusterer = ImageBasedClusterer<LinearImageLabeler<>>;
    using ClustererPtr = std::shared_ptr<Clusterer>;
    using ProjectionParamsPtr = std::unique_ptr<ProjectionParams>;

public:
    explicit Segment(const string& calibration,
            int gnd_angle_tol = 9,
            int ins_angle_tol = 10,
            float ins_dist_tol = 0.1,
            int min_cluster_size = 20,
            int max_cluster_size = 100000,
            int smooth_window_size = 5) {
        _proj_param_ptr = ProjectionParams::FromString(calibration);
        auto gnd_angle_tol_rad = Radians::FromDegrees(static_cast<float>(gnd_angle_tol));
        auto ins_angle_tol_rad = Radians::FromDegrees(static_cast<float>(ins_angle_tol));

        _ground_remover = std::make_shared<GroundRemover>(*_proj_param_ptr, gnd_angle_tol_rad, smooth_window_size);
        _clusterer = std::make_shared<Clusterer>(ins_angle_tol_rad, ins_dist_tol, min_cluster_size, max_cluster_size);
    }
    Segment(const Segment&) = delete;
    Segment& operator=(const Segment&) = delete;
    Segment(Segment &&) = delete;
    Segment& operator=(Segment&&) = delete;
    ~Segment() = default;

    void ExtractGroundAndInstance(cv::Mat depth_image) {
        auto cloud_ptr = Cloud::FromImage(depth_image, *_proj_param_ptr);
        time_utils::Timer timer;
        _ground_remover->OnNewObjectReceived(*cloud_ptr, _no_ground_image);
        cloud_ptr->projection_ptr()->depth_image() = _no_ground_image;
        _clusterer->OnNewObjectReceived(*cloud_ptr, _labels);
        auto current_millis = timer.measure(time_utils::Timer::Units::Milli);
        fprintf(stderr, "INFO: It took %lu ms to process and show everything.\n",
                current_millis);
    }

    cv::Mat GetNoGroundImage() {
        return _no_ground_image;
    }

    cv::Mat GetInstances() {
        return _labels;
    }

private:
    GroundRemoverPtr _ground_remover;
    ClustererPtr _clusterer;
    ProjectionParamsPtr _proj_param_ptr;
    cv::Mat _no_ground_image;
    cv::Mat _labels;
};

namespace py = pybind11;
PYBIND11_MODULE(segment, m) {
    NDArrayConverter::init_numpy();

    py::class_<Segment>(m, "Segment")
            .def(py::init<const string &, int, int, float, int, int, int>(),
                 py::arg("path"),
                 py::arg("ground_angle_tolerance") = 9,
                 py::arg("instance_angle_tolerance") = 10,
                 py::arg("instance_distance_tolerance") = 0.1,
                 py::arg("min_cluster_size") = 20,
                 py::arg("max_cluster_size") = 100000,
                 py::arg("smooth_window_size") = 5)
            .def("extract_ground_and_instance", &Segment::ExtractGroundAndInstance, py::arg("range_image"))
            .def("get_no_ground_image", &Segment::GetNoGroundImage)
            .def("get_instances", &Segment::GetInstances);
}