// Copyright 2022 Takeshi Ishita

#ifndef INCLUDE_EIGEN_HPP_
#define INCLUDE_EIGEN_HPP_

#include <Eigen/Core>

#include <vector>

#include <opencv2/opencv.hpp>


Eigen::Vector2d point2f_to_eigen(const cv::Point2f & p);
Eigen::MatrixXd point2f_to_eigen(const std::vector<cv::Point2f> & keypoints);

#endif  // INCLUDE_EIGEN_HPP_
