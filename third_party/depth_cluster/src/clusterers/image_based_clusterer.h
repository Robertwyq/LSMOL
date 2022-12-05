// Copyright (C) 2017  I. Bogoslavskyi, C. Stachniss, University of Bonn

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.

// You should have received a copy of the GNU General Public License along
// with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef SRC_CLUSTERERS_IMAGE_BASED_CLUSTERER_H_
#define SRC_CLUSTERERS_IMAGE_BASED_CLUSTERER_H_

#include <chrono>
#include <ctime>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/cloud.h"
#include "utils/radians.h"
#include "utils/timer.h"
#include "utils/useful_typedefs.h"

#include "clusterers/abstract_clusterer.h"
#include "image_labelers/diff_helpers/diff_factory.h"
#include "image_labelers/linear_image_labeler.h"
#include "projections/cloud_projection.h"

namespace depth_clustering {

/**
 * @brief      Class for image based clusterer.
 *
 * @tparam     LabelerT  A Labeler class to be used for labeling.
 */
template <typename LabelerT>
class ImageBasedClusterer : public AbstractClusterer {
 public:
  /**
   * @brief      Construct an image-based clusterer.
   *
   * @param[in]  angle_tolerance  The angle tollerance to separate objects
   * @param[in]  min_cluster_size  The minimum cluster size to send
   * @param[in]  max_cluster_size  The maximum cluster size to send
   */
  explicit ImageBasedClusterer(Radians angle_tolerance = 8_deg,
                               float dist_tolerance = 0.1f,
                               uint16_t min_cluster_size = 100,
                               uint16_t max_cluster_size = 25000)
      : AbstractClusterer(0.0, min_cluster_size, max_cluster_size),
        _counter(0),
        _angle_tolerance(angle_tolerance),
        _dist_tolerance(dist_tolerance),
        _diff_types{DiffFactory::DiffType::ANGLES, DiffFactory::DiffType::EUCLIDEAN_DIST} {}

  ~ImageBasedClusterer() override = default;

  /**
   * @brief      Gets called when clusterer receives a cloud to cluster
   *
   * @param[in]  cloud      The cloud to cluster
   * @param[in]  sender_id  The sender identifier
   */
  void OnNewObjectReceived(const Cloud& cloud, cv::Mat& labels) {
    // generate a projection from a point cloud
    if (!cloud.projection_ptr()) {
      fprintf(stderr, "ERROR: projection not initialized in cloud.\n");
      fprintf(stderr, "INFO: cannot label this cloud.\n");
      return;
    }
    time_utils::Timer timer;
    LabelerT image_labeler(cloud.projection_ptr()->depth_image(),
                           cloud.projection_ptr()->params(), _angle_tolerance);
    image_labeler.ComputeLabels_MultiDiff(_diff_types, std::vector<float>({_angle_tolerance.val(), _dist_tolerance}));
    labels = *image_labeler.GetLabelImage();
    fprintf(stderr, "INFO: image based labeling took: %lu us\n",
            timer.measure());
  }

 private:
  int _counter;
  Radians _angle_tolerance;
  float _dist_tolerance;
  std::vector<DiffFactory::DiffType> _diff_types;
};

}  // namespace depth_clustering

#endif  // SRC_CLUSTERERS_IMAGE_BASED_CLUSTERER_H_
