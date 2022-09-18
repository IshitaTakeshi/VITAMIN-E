// Copyright 2022 Takeshi Ishita

#ifndef INCLUDE_CURVATURE_HPP_
#define INCLUDE_CURVATURE_HPP_

#include <Eigen/Core>

inline double dx(const Eigen::MatrixXd & I, const int x, const int y) {
  return 0.5 * (I(y, x+1) - I(y, x-1));
}

inline double dy(const Eigen::MatrixXd & I, const int x, const int y) {
  return 0.5 * (I(y+1, x) - I(y-1, x));
}

inline double dxx(const Eigen::MatrixXd & I, const int x, const int y) {
  // (f(x+1, y) - f(x, y)) - (f(x, y) - f(x-1, y))
  // = f(x+1, y) - 2 * f(x, y) + f(x-1, y)
  return I(y, x+1) - 2.0 * I(y, x) + I(y, x-1);
}

inline double dyy(const Eigen::MatrixXd & I, const int x, const int y) {
  // (f(x, y+1) - f(x, y)) - (f(x, y) - f(x, y-1))
  // = f(x, y+1) - 2 * f(x, y) - f(x, y-1)
  return I(y+1, x) - 2.0 * I(y, x) + I(y-1, x);
}

inline double dxdy(const Eigen::MatrixXd & I, const int x, const int y) {
  return I(y+1, x+1) - I(y+1, x-1) - I(y-1, x+1) + I(y-1, x-1);
}

inline double compute_curvature(
  const double fx,
  const double fy,
  const double fxx,
  const double fyy,
  const double fxfy) {
  return fy * fy * fxx - 2 * fx * fy * fxfy + fx * fx * fyy;
}

inline double compute_curvature(
  const Eigen::MatrixXd & image,
  const int x,
  const int y) {
  const double fx = dx(image, x, y);
  const double fy = dy(image, x, y);
  const double fxx = dxx(image, x, y);
  const double fyy = dyy(image, x, y);
  const double fxfy = dxdy(image, x, y);
  return compute_curvature(fx, fy, fxx, fyy, fxfy);
}

Eigen::MatrixXd curvature_extrema(const Eigen::MatrixXd & image);

#endif  // INCLUDE_CURVATURE_HPP_
