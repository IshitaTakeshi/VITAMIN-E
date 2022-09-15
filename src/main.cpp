#include "curvature.hpp"

#include <iostream>

#include <Eigen/Core>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>


int main()
{
  const cv::Mat cvimage1 = cv::imread("einstein_1/rgb/5390.470225.png", cv::IMREAD_GRAYSCALE);
  const cv::Mat cvimage2 = cv::imread("einstein_1/rgb/5390.507088.png", cv::IMREAD_GRAYSCALE);

  std::cout << "image1.type = " << cvimage1.type() << std::endl;
  std::cout << "image2.type = " << cvimage1.type() << std::endl;

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
