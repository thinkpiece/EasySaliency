/** @file scalespace.cc */
#include "scalespace.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <cassert>

namespace cv
{

// generates the gaussian pyramid
void ScaleSpace::Compute(const Mat& image)
{
// replacing with buildPyramid() opencv-built-in function.
//  buildPyramid(image, pyramid_, levels_, BORDER_REFLECT_101);

  // push initial floor: prologue
  pyramid_.push_back(image);

  // body
  for (int i=1; i<levels_+1; ++i) {
    Mat next_image( Size((pyramid_[i-1].cols+1)/2, (pyramid_[i-1].rows+1)/2),
                   CV_32FC1);
    pyrDown( pyramid_[i-1], next_image, next_image.size(), BORDER_REFLECT_101);
    // downsampling needed? needed.
    pyramid_.push_back(next_image);
  }
}

Mat ScaleSpace::Rescale(const Mat& image) const
{
  double minvalue, maxvalue;
  minMaxLoc(image, &minvalue, &maxvalue);

  Mat converted_image;
  image.convertTo(converted_image, CV_32FC1, 1.0/(maxvalue-minvalue),
                  -minvalue*1.0/(maxvalue-minvalue));

  return converted_image;
}

Mat ScaleSpace::Difference(const int idx1, const int idx2) const
{
  assert(idx1 <= idx2);

  // NOTE:
  // DO NOT interploate the smaller image to the larger one!!
  // the difference between different scales should be matched with the smaller
  // one. Otherwise, the interpolation artifact ruins the difference reponse.
  Mat resample_image(pyramid_[idx2].size(), CV_32FC1);
  resize(pyramid_[idx1], resample_image, resample_image.size(),
         0, 0, INTER_AREA);

  Mat diff(pyramid_[idx2].size(), CV_32FC1);

//  assert(resample_image.size() == diff.size());
  Mat rescaled_a = Rescale(resample_image);
  Mat rescaled_b = Rescale(pyramid_[idx2]);
  absdiff(rescaled_a, rescaled_b, diff);

//  absdiff(resample_image, pyramid_[idx2], diff);
  return diff;
}

}