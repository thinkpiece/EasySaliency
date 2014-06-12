#include "imgdump.h"

#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat ImgDump::Rescale(const cv::Mat& image) const
{
  double minvalue, maxvalue;
  cv::minMaxLoc(image, &minvalue, &maxvalue);

  cv::Mat converted_image;
  image.convertTo(converted_image, CV_8UC1, 255.0/(maxvalue-minvalue),
                  -minvalue*255.0/(maxvalue-minvalue));

  return converted_image;
}

void ImgDump::Export(const cv::Mat& image, bool rescale)
{
  cv::Mat bedumped;

  if (rescale) bedumped = Rescale(image);
  else bedumped = image;

  std::stringstream filename;
  filename << prefix_ << index_++ << ".jpg";
  cv::imwrite( filename.str(), bedumped );
}

void ImgDump::Export(const std::vector<cv::Mat>& list, bool rescale)
{
  const int size = list.size();

  for (int i=0; i<size; ++i) {
    Export(list[i], rescale);
  }
}