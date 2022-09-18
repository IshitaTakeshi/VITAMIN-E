// Copyright 2022 Takeshi Ishita

#include "curvature.hpp"
#include <Eigen/Core>

Eigen::MatrixXd curvature_extrema(const Eigen::MatrixXd & image) {
  Eigen::MatrixXd curvature(image.rows()-2, image.cols()-2);

  for (int y = 1; y < image.rows()-1; y++) {
    for (int x = 1; x < image.cols()-1; x++) {
      const double fx = dx(image, x, y);
      const double fy = dy(image, x, y);
      const double fxx = dxx(image, x, y);
      const double fyy = dyy(image, x, y);
      const double fxfy = dxdy(image, x, y);
      curvature(y-1, x-1) = compute_curvature(fx, fy, fxx, fyy, fxfy);
    }
  }
  return curvature;
}
