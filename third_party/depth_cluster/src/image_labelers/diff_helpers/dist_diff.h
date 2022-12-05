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

#ifndef SRC_IMAGE_LABELERS_DIFF_HELPERS_DIST_DIFF_H_
#define SRC_IMAGE_LABELERS_DIFF_HELPERS_DIST_DIFF_H_

#include <math.h>
#include <algorithm>
#include <vector>

#include "image_labelers/diff_helpers/abstract_diff.h"

namespace depth_clustering {

/**
 * @brief      Class for angle difference.
 */
class DistDiff : public AbstractDiff {
 public:
  /**
   * @brief      Precompute the angles to avoid losing time on that.
   *
   * @param[in]  source_image  The source image
   * @param[in]  params        The projection parameters
   */
  DistDiff(const cv::Mat* source_image, const ProjectionParams* params);

  /**
   * @brief      Compute angle-based difference. See paper for details.
   *
   * @param[in]  from  Pixel from which to compute difference
   * @param[in]  to    Pixel to which to compute difference
   *
   * @return     Angle difference between the values
   */
  float DiffAt(const PixelCoord& from, const PixelCoord& to) const override;

  /**
   * @brief      Threshold is satisfied if dist is BIGGER than threshold
   */
  inline bool SatisfiesThreshold(float dist, float threshold) const override {
    return dist < threshold;
  }

  /**
   * @brief      Visualize \f$\beta\f$ angles as a `cv::Mat` color image.
   *
   * @return     `cv::Mat` color image with red channel showing \f$\beta\f$
   *              angles in row direction and green channel in col direction.
   */
  cv::Mat Visualize() const override { return cv::Mat(); }

 private:
  /**
   * @brief      Pre-compute values for angles for all cols and rows
   */
  void PreComputeAlphaVecs();

  /**
   * @brief      Calculates the alpha.
   *
   * @param[in]  current   The current
   * @param[in]  neighbor  The neighbor
   *
   * @return     The alpha.
   */
  float ComputeAlpha(const PixelCoord& current,
                     const PixelCoord& neighbor) const;

  const ProjectionParams* _params = nullptr;
  std::vector<float> _row_alphas;
  std::vector<float> _col_alphas;
};

}  // namespace depth_clustering

#endif  // SRC_IMAGE_LABELERS_DIFF_HELPERS_DIST_DIFF_H_
