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

#ifndef SRC_CLUSTERERS_IMAGE_BASED_CLUSTERER_v2_H_
#define SRC_CLUSTERERS_IMAGE_BASED_CLUSTERER_v2_H_

#include <chrono>
#include <ctime>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "communication/abstract_client.h"
#include "communication/abstract_sender.h"
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
class ImageBasedClusterer_v2 : public AbstractClusterer {
 public:
  using Receiver = AbstractClient<Cloud>;
  using Sender = AbstractSender<std::unordered_map<uint16_t, Cloud>>;

  /**
   * @brief      Construct an image-based clusterer.
   *
   * @param[in]  angle_tollerance  The angle tollerance to separate objects
   * @param[in]  dist_tollerance  The dist tollerance to separate objects
   * @param[in]  min_cluster_size  The minimum cluster size to send
   * @param[in]  max_cluster_size  The maximum cluster size to send
   */
  explicit ImageBasedClusterer_v2(Radians angle_tollerance = 8_deg,
                               float dist_tollerance = 0.1f,
                               uint16_t min_cluster_size = 100,
                               uint16_t max_cluster_size = 25000)
      : AbstractClusterer(0.0, min_cluster_size, max_cluster_size),
        _counter(0),
        _angle_tollerance(angle_tollerance),
        _dist_tollerance(dist_tollerance),
        _label_client{nullptr} {}

  virtual ~ImageBasedClusterer_v2() {}

  /**
   * @brief      Sets the label image client.
   *
   * @param      client  The client to receive color images with labels
   */
  void SetLabelImageClient(AbstractClient<cv::Mat>* client) {
    this->_label_client = client;
  }

  /**
   * @brief      Gets called when clusterer receives a cloud to cluster
   *
   * @param[in]  cloud      The cloud to cluster
   * @param[in]  sender_id  The sender identifier
   */
  void OnNewObjectReceived(const Cloud& cloud, const int sender_id) override {
    // generate a projection from a point cloud
    if (!cloud.projection_ptr()) {
      fprintf(stderr, "ERROR: projection not initialized in cloud.\n");
      fprintf(stderr, "INFO: cannot label this cloud.\n");
      return;
    }
    time_utils::Timer timer;
    LabelerT image_labeler(cloud.projection_ptr()->depth_image(),
                           cloud.projection_ptr()->params(), _angle_tollerance);

    _diff_types.push_back(DiffFactory::DiffType::ANGLES);
    _diff_types.push_back(DiffFactory::DiffType::EUCLIDEAN_DIST);
    image_labeler.ComputeLabels_MultiDiff(_diff_types, std::vector<float>({_angle_tollerance.val(), _dist_tollerance}));
    const cv::Mat* labels_ptr = image_labeler.GetLabelImage();
    fprintf(stderr, "INFO: image based labeling took: %lu us\n",
            timer.measure());

    // send image to whoever wants to get it
    if (_label_client) {
      _label_client->OnNewObjectReceived(*labels_ptr, this->id());
    }

    fprintf(stderr, "INFO: labels image sent to clients in: %lu us\n",
            timer.measure());

    // create 3d clusters from image labels
    std::unordered_map<uint16_t, Cloud> clusters;
    for (int row = 0; row < labels_ptr->rows; ++row) {
      for (int col = 0; col < labels_ptr->cols; ++col) {
        const auto& point_container = cloud.projection_ptr()->at(row, col);
        if (point_container.IsEmpty()) {
          // this is ok, just continue, nothing interesting here, no points.
          continue;
        }
        uint16_t label = labels_ptr->at<uint16_t>(row, col);
        if (label < 1) {
          // this is a default label, skip
          continue;
        }
        for (const auto& point_idx : point_container.points()) {
          const auto& point = cloud.points()[point_idx];
          clusters[label].push_back(point);
        }
      }
    }

    // filter out unfitting clusters
    std::vector<uint16_t> labels_to_erase;
    for (const auto& kv : clusters) {
      const auto& cluster = kv.second;
      if (cluster.size() < this->_min_cluster_size ||
          cluster.size() > this->_max_cluster_size) {
        labels_to_erase.push_back(kv.first);
      }
    }
    for (auto label : labels_to_erase) {
      clusters.erase(label);
    }

    fprintf(stderr, "INFO: prepared clusters in: %lu us\n", timer.measure());

    this->ShareDataWithAllClients(clusters);
    fprintf(stderr, "INFO: clusters shared: %lu us\n", timer.measure());
  }

 private:
  int _counter;
  Radians _angle_tollerance;
  float _dist_tollerance;

  AbstractClient<cv::Mat>* _label_client;

  std::vector<DiffFactory::DiffType> _diff_types;
};

}  // namespace depth_clustering

#endif  // SRC_CLUSTERERS_IMAGE_BASED_CLUSTERER_v2_H_
