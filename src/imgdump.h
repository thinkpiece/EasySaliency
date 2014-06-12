/**
 *  @file   imgdump.h
 *  @author Junyoung Park (thinkpiece@yahoo.com)
 *  @brief  exports the images for debug & academic purposes
 */
#include <opencv2/core/core.hpp>
#include <vector>
#include <string>

class ImgDump
{
 public:
  ImgDump() : index_(0) { }
  ImgDump( std::string prefix ) : index_(0) { prefix_ = prefix; }

  void Export(const cv::Mat& image, bool rescale);
  void Export(const std::vector<cv::Mat>& list, bool rescale);

 private:
  cv::Mat Rescale(const cv::Mat& image) const;

 private:
  std::string prefix_; // filename prefix
  int index_;          // filename index
};
