// Copyright 2022 Takeshi Ishita

#include "keypoint.hpp"

#include <vector>

std::tuple<std::vector<cv::KeyPoint>, cv::Mat> extract(const cv::Mat & image) {
  const cv::Ptr<cv::BRISK> extractor = cv::BRISK::create();
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  extractor->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
  return std::make_tuple(keypoints, descriptors);
}

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
