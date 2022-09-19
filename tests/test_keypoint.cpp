// Copyright 2022 Takeshi Ishita

#include <gmock/gmock.h>

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>

#include "keypoint.hpp"


TEST(ExtractAndMatch, SmokeTest) {
  const cv::Mat image = cv::imread("../data/flower.jpg");
  EXPECT_FALSE(image.empty());

  const auto [keypoints1, descriptors1] = extract(image);
  const auto [keypoints2, descriptors2] = extract(image);
  const auto matches = match(descriptors1, descriptors2);

  for (size_t i = 0; i < matches.size(); i++) {
    EXPECT_EQ(matches[i][0].queryIdx, matches[i][0].trainIdx);
    EXPECT_EQ(matches[i][0].queryIdx, matches[i][1].queryIdx);
    EXPECT_EQ(matches[i][0].distance, 0);
  }
}

TEST(Matching, GoodMatchIndices) {
  const double distance = 2.;
  const double ratio = 0.4;

  const int i = 0;

  const std::vector<std::vector<cv::DMatch>> matches = {
    {cv::DMatch(i, i, distance * ratio - 0.2), cv::DMatch(i, i, distance)},
    {cv::DMatch(i, i, distance * ratio - 0.1), cv::DMatch(i, i, distance)},
    {cv::DMatch(i, i, distance * ratio - 0.0), cv::DMatch(i, i, distance)},
    {cv::DMatch(i, i, distance * ratio + 0.1), cv::DMatch(i, i, distance)},
    {cv::DMatch(i, i, distance * ratio + 0.2), cv::DMatch(i, i, distance)}
  };

  EXPECT_THAT(good_match_indices(matches, ratio), testing::ElementsAre(0, 1));
}

TEST(GetMatch0, SmokeTest) {
  const double distance = 1.;

  const std::vector<std::vector<cv::DMatch>> matches = {
    {cv::DMatch(1, 2, distance), cv::DMatch(2, 3, distance)},
    {cv::DMatch(1, 3, distance), cv::DMatch(2, 4, distance)},
    {cv::DMatch(1, 4, distance), cv::DMatch(2, 5, distance)}
  };

  const std::vector<cv::DMatch> matches0 = get_match0(matches);

  EXPECT_EQ(matches0.size(), 3);

  EXPECT_EQ(matches0[0].queryIdx, 1);
  EXPECT_EQ(matches0[0].trainIdx, 2);

  EXPECT_EQ(matches0[1].queryIdx, 1);
  EXPECT_EQ(matches0[1].trainIdx, 3);

  EXPECT_EQ(matches0[2].queryIdx, 1);
  EXPECT_EQ(matches0[2].trainIdx, 4);
}

TEST(AsIntegers, SmokeTest) {
  const double distance = 1.;

  const std::vector<cv::DMatch> matches12 = {
    cv::DMatch(0, 2, distance),
    cv::DMatch(1, 3, distance),
    cv::DMatch(4, 1, distance)
  };

  const auto [indices1, indices2] = as_integers(matches12);

  EXPECT_THAT(indices1, testing::ElementsAre(0, 1, 4));
  EXPECT_THAT(indices2, testing::ElementsAre(2, 3, 1));
}
