// Copyright 2022 Takeshi Ishita

#ifndef INCLUDE_MATCHING_HPP_
#define INCLUDE_MATCHING_HPP_

#include <tuple>
#include <vector>
#include <opencv2/opencv.hpp>

std::tuple<std::vector<cv::KeyPoint>, cv::Mat> extract(const cv::Mat & image);

std::vector<std::vector<cv::DMatch>> match(
  const cv::Mat & descriptors1,
  const cv::Mat & descriptors2);

std::vector<cv::DMatch> filter_by_match_distance(
  const std::vector<std::vector<cv::DMatch>> & matches12,
  const double distance_ratio);

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
matched_keypoints(
  const std::vector<cv::KeyPoint> & keypoints1,
  const std::vector<cv::KeyPoint> & keypoints2,
  const std::vector<cv::DMatch> & matches12);

#endif  // INCLUDE_MATCHING_HPP_
