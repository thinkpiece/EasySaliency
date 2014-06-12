/**
 *  @file demo.cc
 *  @author Junyoung Park (thinkpiece@yahoo.com)
 *  @brief An example on how to use the EasySaliency library
 *
 */
#include "esconfig.h"
#include "saliency.h"

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void help( char** argv )
{
  std::cout << "EasySaliency demo "
            << "v" << ESLIB_VERSION_MAJOR << "." << ESLIB_VERSION_MINOR
            << "\nUsage: " << argv[0]
            << "[path/to/input_image] [path/to/output_image] \n"
            << "\nNote: Proper result when the image is larger than 640x480"
            << std::endl;
}

int main( int argc, char **argv )
{
  using namespace cv;

  if (argc != 3) {
    help(argv);
    return -1;
  }

  Mat input_image = imread(argv[1]);

  if ( !input_image.data ) {
    std::cout << "Error reading image " << argv[1] << std::endl;
    return -1;
  }

  SaliencyMap test;

  test.Compute(input_image);

  // print out input image info.
  std::cout << "input image size: (" << input_image.cols << ","
            << input_image.rows << ")" << std::endl;

  // print out saliency map info.
  std::cout << "saliency map size: (" << test.get_saliency_map().cols
             << "," << test.get_saliency_map().rows << ")" << std::endl;

  imwrite(argv[2], test.get_saliency_map());

  return 0;
}