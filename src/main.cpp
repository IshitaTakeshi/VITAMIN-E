// Copyright 2022 Takeshi Ishita

#include <Eigen/Core>

#include <iostream>

#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "curvature.hpp"
#include "matching.hpp"

std::vector<std::vector<cv::DMatch>> match(
  const cv::Mat & descriptors1,
  const cv::Mat & descriptors2) {
  std::vector<std::vector<cv::DMatch>> matches;
  const cv::BFMatcher matcher(cv::NORM_HAMMING);
  matcher.knnMatch(descriptors1, descriptors2, matches, 2);
  return matches;
}

std::vector<cv::DMatch> filter_by_match_distance(
  const std::vector<std::vector<cv::DMatch>> & matches12,
  const double distance_ratio) {
  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < matches12.size(); ++i) {
      const double d1 = matches12[i][0].distance;
      const double d2 = matches12[i][1].distance;
      if (d1 <= d2 * distance_ratio) {
          good_matches.push_back(matches12[i][0]);
      }
  }
  return good_matches;
}

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
matched_keypoints(
  const std::vector<cv::KeyPoint> & keypoints1,
  const std::vector<cv::KeyPoint> & keypoints2,
  const std::vector<cv::DMatch> & matches12) {
  std::vector<cv::Point2f> matched1;
  std::vector<cv::Point2f> matched2;
  for (size_t i = 0; i < matches12.size(); i++) {
    matched1.push_back(keypoints1[matches12[i].queryIdx].pt);
    matched2.push_back(keypoints2[matches12[i].trainIdx].pt);
  }
  return std::make_tuple(matched1, matched2);
}

std::tuple<std::vector<cv::KeyPoint>, cv::Mat> extract(const cv::Mat & image) {
  const cv::Ptr<cv::BRISK> extractor = cv::BRISK::create();
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  extractor->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
  return std::make_tuple(keypoints, descriptors);
}

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
  const auto good_matches12 = filter_by_match_distance(matches12, 0.75);
  std::cout << "matches12.size() == " << matches12.size() << std::endl;
  std::cout << "good_matches12.size() == "
            << good_matches12.size() << std::endl;
  const auto [matched1, matched2] = matched_keypoints(
    keypoints1, keypoints2, good_matches12);

  cv::Mat image_matches;
  cv::drawMatches(
      cvimage1, keypoints1, cvimage2, keypoints2, good_matches12, image_matches,
      cv::Scalar::all(-1), cv::Scalar::all(-1),
      std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  cv::imshow("Good Matches & Object detection", image_matches);

  const cv::Mat H = cv::findHomography(matched1, matched2, cv::RANSAC);

  std::vector<cv::Point2f> obj_corners(4);
  obj_corners[0] = cv::Point2f(0, 0);
  obj_corners[1] = cv::Point2f( (float)cvimage1.cols, 0 );
  obj_corners[2] = cv::Point2f( (float)cvimage1.cols, (float)cvimage1.rows );
  obj_corners[3] = cv::Point2f( 0, (float)cvimage1.rows );
  std::vector<cv::Point2f> scene_corners(4);
  perspectiveTransform(obj_corners, scene_corners, H);

  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  cv::line(
      image_matches, scene_corners[0] + cv::Point2f((float)cvimage1.cols, 0),
      scene_corners[1] + cv::Point2f((float)cvimage1.cols, 0), cv::Scalar(0, 255, 0), 4);
  cv::line(
      image_matches, scene_corners[1] + cv::Point2f((float)cvimage1.cols, 0),
      scene_corners[2] + cv::Point2f((float)cvimage1.cols, 0), cv::Scalar( 0, 255, 0), 4);
  cv::line(
      image_matches, scene_corners[2] + cv::Point2f((float)cvimage1.cols, 0),
      scene_corners[3] + cv::Point2f((float)cvimage1.cols, 0), cv::Scalar( 0, 255, 0), 4);
  cv::line(
      image_matches, scene_corners[3] + cv::Point2f((float)cvimage1.cols, 0),
      scene_corners[0] + cv::Point2f((float)cvimage1.cols, 0), cv::Scalar( 0, 255, 0), 4);
  //-- Show detected matches
  cv::imshow("Good Matches & Object detection", image_matches);

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
