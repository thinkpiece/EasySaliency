/**
 *  @file   scalespace.h
 *  @author Junyoung Park (thinkpiece@yahoo.com)
 *  @brief  scale-space pyramid class
 *
 *  Nine spatial scales are created using dyadic Gaussian pyramids
 *  yielding progressively pass filter and subsample the input image.
 *  See below for details:
 *
 *  H. Greenspan, et al., "Overcomplete steerable pyramid filters and rotation
 *  invariance," Proc. IEEE CVPR, 1994.
 *
 */
#ifndef __EASY_SALIENCY_SCALESPACE_H__
#define __EASY_SALIENCY_SCALESPACE_H__

#include <vector>
#include <opencv2/core/core.hpp>

namespace cv
{

/** generates the Gaussian Pyramid (Scale-space) of the input image.
 *  the level of the pyramid is decided by the parameter levels_ and
 *  the first value starts from 1.0
 *
 *  * performs Gaussian filtering [ 1 sqrt(2) 2 2*sqrt(2) 4 ... ] unitl
 *    reaching the 'levels'
 *  * downsampling the original image until reaching the 'levels'
 */
class ScaleSpace
{
 public:
  /** Constructor */
  ScaleSpace(const int levels)
      : levels_(levels), pyramid_()
  { }

  /** Constructor with base image */
  ScaleSpace(const int levels, const Mat& image) : levels_(levels)
  { Compute(image); }

  /** destructor */
  ~ScaleSpace() { }

  /** generates the scale-space based on the image
   *  @return the number of pyramid level
   */
  void Compute(const Mat& image);

  /** generates the center-surround difference */
  Mat Difference(const int idx1, const int idx2) const;

  /** return the pyramid image */
  Mat get_image(const int index) const { return pyramid_[index]; }

  Size get_size(const int index) const { return pyramid_[index].size(); }
  std::vector<Mat> get_pyramid() const { return pyramid_; }

 // private functions for internal uses
 private:
  Mat Rescale(const Mat& image) const;

 private:
  int levels_;
  std::vector<Mat> pyramid_;
};

}

#endif
