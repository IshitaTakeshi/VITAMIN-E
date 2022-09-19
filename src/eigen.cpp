// Copyright 2022 Takeshi Ishita

#include "eigen.hpp"


Eigen::Vector2d point2f_to_eigen(const cv::Point2f & p) {
  return Eigen::Vector2d(p.x, p.y);
}

Eigen::MatrixXd point2f_to_eigen(const std::vector<cv::Point2f> & keypoints) {
  Eigen::MatrixXd matrix(2, keypoints.size());
  for (size_t i = 0; i < keypoints.size(); i++) {
    matrix.col(i) = point2f_to_eigen(keypoints[i]);
  }
  return matrix;
}
