// Copyright 2022 Takeshi Ishita

#include <Eigen/Core>

#include <iostream>

#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "curvature.hpp"
#include "keypoint.hpp"
#include "vector.hpp"

int main() {
  const std::string filename0 = "einstein_1/rgb/5390.470225.png";
  const std::string filename1 = "einstein_1/rgb/5390.875722.png";
  const cv::Mat cvimage1 = cv::imread(filename0, cv::IMREAD_GRAYSCALE);
  const cv::Mat cvimage2 = cv::imread(filename1, cv::IMREAD_GRAYSCALE);

  std::cout << "image1.type = " << cvimage1.type() << std::endl;
  std::cout << "image2.type = " << cvimage1.type() << std::endl;

  const auto [keypoints1, descriptors1] = extract(cvimage1);
  const auto [keypoints2, descriptors2] = extract(cvimage2);

  const auto matches12 = match(descriptors1, descriptors2);
  const auto good_indices = good_match_indices(matches12, 0.75);
  const auto matches1 = get_match0(matches12);
  const auto good_matches1 = get_by_indices(matches1, good_indices);
  const auto [indices1, indices2] = as_integers(good_matches1);
  const auto matched1 = get_as_references(keypoints1, indices1);
  const auto matched2 = get_as_references(keypoints2, indices2);

  cv::Mat image_matches;
  cv::drawMatches(
      cvimage1, keypoints1, cvimage2, keypoints2,
      good_matches1, image_matches,
      cv::Scalar::all(-1), cv::Scalar::all(-1),
      std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  cv::imshow("Good Matches & Object detection", image_matches);

  // const cv::Mat H = cv::findHomography(matched1, matched2, cv::RANSAC);

  Eigen::MatrixXd image1, image2;
  cv::cv2eigen(cvimage1, image1);
  cv::cv2eigen(cvimage2, image2);

  const Eigen::MatrixXd curvature1 = curvature_extrema(image1);
  const Eigen::MatrixXd curvature2 = curvature_extrema(image2);

  cv::Mat cvcurvature1, cvcurvature2;
  cv::eigen2cv(curvature1, cvcurvature1);
  cv::eigen2cv(curvature2, cvcurvature2);

  cv::imshow("curvature1", cvcurvature1);
  cv::imshow("curvature2", cvcurvature2);
  cv::waitKey(0);

  return 0;
}
