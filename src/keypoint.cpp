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

std::vector<int> good_match_indices(
  const std::vector<std::vector<cv::DMatch>> & matches,
  const double distance_ratio) {
  std::vector<int> indices;
  for (size_t i = 0; i < matches.size(); ++i) {
    const double d0 = matches[i][0].distance;
    const double d1 = matches[i][1].distance;
    if (d0 < d1 * distance_ratio) {
      indices.push_back(i);
    }
  }
  return indices;
}

std::vector<cv::DMatch> get_match0(
  const std::vector<std::vector<cv::DMatch>> & matches) {
  std::vector<cv::DMatch> m;
  for (size_t i = 0; i < matches.size(); i++) {
    m.push_back(matches[i][0]);
  }
  return m;
}

std::pair<std::vector<int>, std::vector<int>> as_integers(
  const std::vector<cv::DMatch> & matches) {
  std::vector<int> indices1;
  std::vector<int> indices2;
  for (const auto & m : matches) {
    indices1.push_back(m.queryIdx);
    indices2.push_back(m.trainIdx);
  }
  return std::make_pair(indices1, indices2);
}
