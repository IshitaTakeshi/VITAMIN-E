// Copyright 2022 Takeshi Ishita

#include <gtest/gtest.h>

#include "curvature.hpp"

TEST(Curvature, Derivatives) {
  Eigen::Matrix3d I;
  I <<
    2, 6, 3,
    5, 4, 2,
    0, 1, 1;

  EXPECT_EQ(dx(I, 1, 1), 0.5 * (2-5));
  EXPECT_EQ(dy(I, 1, 1), 0.5 * (1-6));
  EXPECT_EQ(dxx(I, 1, 1), 2 - 2 * 4 + 5);
  EXPECT_EQ(dyy(I, 1, 1), 1 - 2 * 4 + 6);
  EXPECT_EQ(dxdy(I, 1, 1), 1 - 0 - 3 + 2);
}


TEST(Curvature, ComputeCurvature) {
  const double fx = 2.;
  const double fy = 3.;
  const double fxx = 4.;
  const double fyy = 5.;
  const double fxfy = 6.;
  EXPECT_EQ(
    compute_curvature(fx, fy, fxx, fyy, fxfy),
    fy * fy * fxx - 2 * fx * fy * fxfy + fx * fx * fyy);
}

TEST(Curvature, Extrema) {
  Eigen::MatrixXd I(4, 5);
  I <<
    2, 6, 3, 5, 4,
    5, 4, 2, 3, 1,
    0, 1, 1, 3, 5,
    4, 3, 2, 5, 1;

  const Eigen::MatrixXd C = curvature_extrema(I);
  EXPECT_EQ(C.rows(), 2);
  EXPECT_EQ(C.cols(), 3);

  EXPECT_EQ(C(0, 0), compute_curvature(I, 1, 1));
  EXPECT_EQ(C(0, 1), compute_curvature(I, 2, 1));
  EXPECT_EQ(C(0, 2), compute_curvature(I, 3, 1));
  EXPECT_EQ(C(1, 0), compute_curvature(I, 1, 2));
  EXPECT_EQ(C(1, 1), compute_curvature(I, 2, 2));
  EXPECT_EQ(C(1, 2), compute_curvature(I, 3, 2));
}
