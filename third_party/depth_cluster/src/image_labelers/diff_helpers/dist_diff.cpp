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

#include <math.h>
#include <vector>
#include <algorithm>

#include "image_labelers/diff_helpers/dist_diff.h"

namespace depth_clustering {

DistDiff::DistDiff(const cv::Mat* source_image,
                     const ProjectionParams* params)
    : AbstractDiff{source_image}, _params{params} {
  PreComputeAlphaVecs();
}

float DistDiff::DiffAt(const PixelCoord& from, const PixelCoord& to) const {
  const float& current_depth = _source_image->at<float>(from.row, from.col);
  const float& neighbor_depth = _source_image->at<float>(to.row, to.col);
  float alpha = ComputeAlpha(from, to);
  if (alpha > _params->h_span().val() - 0.05) {
    // we are over the border
    const float span = _params->h_span().val();
    if (alpha > span) {
      alpha -= span;
    } else {
      alpha = span - alpha;
    }
  }
  float d1 = std::max(current_depth, neighbor_depth);
  float d2 = std::min(current_depth, neighbor_depth);
  float d12 = std::sqrt(d1*d1 + d2*d2 - 2*d1*d2*cos(alpha));
  return d12;
}

void DistDiff::PreComputeAlphaVecs() {
  _row_alphas.reserve(_params->rows());
  for (size_t r = 0; r < _params->rows() - 1; ++r) {
    _row_alphas.push_back(
        fabs((_params->AngleFromRow(r + 1) - _params->AngleFromRow(r)).val()));
  }
  // add last row alpha
  _row_alphas.push_back(0.0f);
  // now handle the cols
  _col_alphas.reserve(_params->cols());
  for (size_t c = 0; c < _params->cols() - 1; ++c) {
    _col_alphas.push_back(
        fabs((_params->AngleFromCol(c + 1) - _params->AngleFromCol(c)).val()));
  }
  // handle last angle where we wrap columns
  float last_alpha = fabs((_params->AngleFromCol(0) -
                           _params->AngleFromCol(_params->cols() - 1)).val());
  last_alpha -= _params->h_span().val();
  _col_alphas.push_back(last_alpha);
}

float DistDiff::ComputeAlpha(const PixelCoord& current,
                              const PixelCoord& neighbor) const {
  if ((current.col == 0 &&
       neighbor.col == static_cast<int>(_params->cols() - 1)) ||
      (neighbor.col == 0 &&
       current.col == static_cast<int>(_params->cols() - 1))) {
    // this means we wrap around
    return _col_alphas.back();
  }
  if (current.row < neighbor.row) {
    return _row_alphas[current.row];
  } else if (current.row > neighbor.row) {
    return _row_alphas[neighbor.row];
  } else if (current.col < neighbor.col) {
    return _col_alphas[current.col];
  } else if (current.col > neighbor.col) {
    return _col_alphas[neighbor.col];
  }
  return 0;
}

}  // namespace depth_clustering
