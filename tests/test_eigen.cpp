// Copyright 2022 Takeshi Ishita

#include <Eigen/Core>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "eigen.hpp"

TEST(Eigen, Point2fToEigenVector) {
  const Eigen::Vector2d p = point2f_to_eigen(cv::Point2f(5., 3.));
  const Eigen::Vector2d expected(5., 3.);
  EXPECT_EQ((p - expected).norm(), 0.);
}

TEST(Eigen, Point2fToEigenMatrix) {
}
