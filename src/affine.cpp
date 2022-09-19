// Copyright 2022 Takeshi Ishita

#include <Eigen/Core>

Eigen::MatrixXd estimate_affine(
  const Eigen::Matrix & keypoints1,
  const Eigen::Matrix & keypoints2) {
  return Eigen::Matrix3d::Identity();
}
