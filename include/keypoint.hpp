// Copyright 2022 Takeshi Ishita

#ifndef INCLUDE_KEYPOINT_HPP_
#define INCLUDE_KEYPOINT_HPP_

#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

std::tuple<std::vector<cv::KeyPoint>, cv::Mat> extract(const cv::Mat & image);

std::vector<std::vector<cv::DMatch>> match(
  const cv::Mat & descriptors1,
  const cv::Mat & descriptors2);

std::vector<int> good_match_indices(
  const std::vector<std::vector<cv::DMatch>> & matches12,
  const double distance_ratio);

std::vector<cv::DMatch> get_match0(
  const std::vector<std::vector<cv::DMatch>> & matches);

std::pair<std::vector<int>, std::vector<int>> as_integers(
  const std::vector<cv::DMatch> & matches);

#endif  // INCLUDE_KEYPOINT_HPP_
