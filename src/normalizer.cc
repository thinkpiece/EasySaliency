/** @file normalizer.cc */
#include "normalizer.h"

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

namespace cv
{

LocalMaxNormalizer* LocalMaxNormalizer::pinst_  = 0;
IterativeNormalizer* IterativeNormalizer::pinst_ = 0;

Mat LocalMaxNormalizer::Compute(const Mat& image)
{
  // 1. normalization & 2. finding the global maximum M
  double minvalue, maxvalue;
  minMaxLoc(image, &minvalue, &maxvalue);

  Mat norm_image;
  image.convertTo(norm_image, CV_32FC1, 1.0/(maxvalue-minvalue),
                  -minvalue*1.0/(maxvalue-minvalue));

  // 3. computing the average ~m of local maxima
  //
  // in order to acquire the local maxima, we just see the neighbors
  // in 8-directions
  int local_max_num   = 0;      // number of local max
  float local_max_sum = 0.0f;   // sum of local max

  const int width  = norm_image.cols;
  const int height = norm_image.rows;

  for (int y=0; y<height; ++y) {
    int top    = y-1;
    int bottom = y+1;

    if (top < 0)          top = 0;
    if (bottom >= height) bottom = height-1;

    for (int x=0; x<width; x++) {
      int left   = x-1;
      int right  = x+1;

      if (left < 0)       left = 0;
      if (right >= width) right = width-1;

      // if local maximum
      if (norm_image.at<float>(y,x) >= norm_image.at<float>(top, left    ) &&
          norm_image.at<float>(y,x) >= norm_image.at<float>(top, x       ) &&
          norm_image.at<float>(y,x) >= norm_image.at<float>(top, right   ) &&
          norm_image.at<float>(y,x) >= norm_image.at<float>(y, left      ) &&
          norm_image.at<float>(y,x) >= norm_image.at<float>(y, right     ) &&
          norm_image.at<float>(y,x) >= norm_image.at<float>(bottom, left ) &&
          norm_image.at<float>(y,x) >= norm_image.at<float>(bottom, x    ) &&
          norm_image.at<float>(y,x) >= norm_image.at<float>(bottom, right)) {
        local_max_num++;
        local_max_sum += norm_image.at<float>(y,x);
      }
    }
  }

  // 3) globally multiplying the map by (M-~m)^2
  float scale_val = 1.0f;

  if (local_max_num > 0)
      scale_val = (1.0f - (local_max_sum/local_max_num)) *
                  (1.0f - (local_max_sum/local_max_num));

  norm_image.mul(scale_val);

  return norm_image;
}

IterativeNormalizer::IterativeNormalizer()
    : inhibition_constant_(0.02f), iteration_(3),
      excitation_sigma_(0.02f), inhibition_sigma_(0.25f),
      excitation_coef_(0.5f), inhibition_coef_(1.5f)
{ }

Mat IterativeNormalizer::Compute(const Mat& image)
{
  // first, normalized to a fixed dynamic range (between 0 and 1)
  // in order to eliminate feature-dependent amplitude differences
  // due to different feature extraction mechanisms.
  double minvalue, maxvalue;
  minMaxLoc(image, &minvalue, &maxvalue);

  Mat norm_image;
  image.convertTo(norm_image, CV_32FC1, 1.0/(maxvalue-minvalue),
                  -minvalue*1.0/(maxvalue-minvalue));

  // If you do not want to use minMaxLoc(),
  // use the following code for finding the min, max
  // MatConstIterator_<float> maxvalue =
  //     std::max_element(image.begin<float>(), image.end<float>());
  // MatConstIterator_<float> minvalue =
  //     std::min_element(image.begin<float>(), image.end<float>());

  // then, iteratively convolved by a large 2-D DoG filter, the original
  // image is added to the result, and negative results are set to zero
  // after each iteration.
  //
  // The DoG filter yields strong local excitation at each visual location,
  // which is counteracted by broad inhibition from neighboring locations.
  int l = std::max(norm_image.rows, norm_image.cols);

  const float excitation_sig = l*excitation_sigma_;
  const float inhibition_sig = l*inhibition_sigma_;

  // excitation kernel
  Mat ekernel = getGaussianKernel(excitation_sig*6+1, excitation_sig, CV_32FC1);
  ekernel.mul(excitation_coef_);
  Ptr<FilterEngine> egauss = createSeparableLinearFilter(CV_32FC1, CV_32FC1,
      ekernel, ekernel, Point(-1,-1), 0, BORDER_REFLECT_101, BORDER_REFLECT_101);

  Mat ikernel = getGaussianKernel(inhibition_coef_*6+1,
                                  inhibition_coef_, CV_32FC1);
  ikernel.mul(inhibition_coef_);
  Ptr<FilterEngine> igauss = createSeparableLinearFilter(CV_32FC1, CV_32FC1,
      ikernel, ikernel, Point(-1,-1), 0, BORDER_REFLECT_101, BORDER_REFLECT_101);

  for(int i=0; i<iteration_; ++i) {
    // excitation and inhibition
    Mat excitation(norm_image.size(), CV_32FC1);
    Mat inhibition(norm_image.size(), CV_32FC1);

    egauss->apply(norm_image, excitation);
    igauss->apply(norm_image, inhibition);

    float g_inhibition_constant_ = inhibition_constant_;
    norm_image += excitation - inhibition - g_inhibition_constant_;
    norm_image = abs(norm_image);
  }

  // resize to [0..1]
  minMaxLoc(norm_image, &minvalue, &maxvalue);
  norm_image.convertTo(norm_image, CV_32FC1, 1.0/(maxvalue-minvalue),
                       -minvalue*1.0/(maxvalue-minvalue));

  return norm_image;
}

}