/** @file saliency.cc */
#include "saliency.h"
#include "scalespace.h"
#include "normalizer.h"

#include "imgdump.h"

// c standard libraries
#include <vector>
#include <iterator>
#include <cmath>
#include <cassert>

// opencv libraries
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace cv
{

// default parameters
SaliencyMap::SaliencyMap(NormalizationMethod norm_method, bool ifdebug)
    : saliency_level_(4),
      center_level_min_(2), center_level_max_(4),
      surround_delta_min_(3), surround_delta_max_(4),
      debug_(ifdebug)
{
  pyramid_level_ = center_level_max_ + surround_delta_max_ + 1;

  switch(norm_method) {
    case kLocalMaxNormalize:  normalizer_ = LocalMaxNormalizer::Create();
                              break;
    case kIterativeNormalize: normalizer_ = IterativeNormalizer::Create();
                              break;
    default:                  normalizer_ = LocalMaxNormalizer::Create();
  }
}

// destructor
SaliencyMap::~SaliencyMap()
{ }

Mat SaliencyMap::ConvertMap(const Mat& image) const
{
  double minvalue, maxvalue;
  minMaxLoc(image, &minvalue, &maxvalue);

  Mat converted_image;
  image.convertTo(converted_image, CV_8UC1, 255.0/(maxvalue-minvalue),
                  -minvalue*255.0/(maxvalue-minvalue));

  return converted_image;
}

Mat SaliencyMap::get_saliency_map() const
{ return ConvertMap(saliency_map_); }

Mat SaliencyMap::get_conspicuity_map_intensity() const
{ return ConvertMap(conspicuity_map_intensity_); }

Mat SaliencyMap::get_conspicuity_map_color() const
{ return ConvertMap(conspicuity_map_color_); }

Mat SaliencyMap::get_conspicuity_map_orientation() const
{ return ConvertMap(conspicuity_map_orientation_); }

void SaliencyMap::set_center_pyramid_level(unsigned int min, unsigned int max)
{
  center_level_min_ = min; center_level_max_ = max;
  pyramid_level_ = center_level_max_ + surround_delta_max_ + 1;
}

void SaliencyMap::set_surround_pyramid_level(unsigned int min, unsigned int max)
{ surround_delta_min_ = min; surround_delta_max_ = max;
  pyramid_level_ = center_level_max_ + surround_delta_max_ + 1;
}

void SaliencyMap::set_saliency_map_level(unsigned int level)
{ saliency_level_ = level; }

// generates the conspicuity maps for intensity
void SaliencyMap::GenerateConspicuity(const std::vector<Mat>& maps,
                                      Mat& conspicuity)
{
  std::vector<Mat>::const_iterator map_tracer;

  conspicuity = Mat::zeros(saliency_map_shape_, CV_32FC1);

  for (map_tracer = maps.begin(); map_tracer != maps.end(); ++map_tracer) {
    Mat normalized_map = normalizer_->Compute(*map_tracer);

    if (normalized_map.size() != saliency_map_shape_) {
      Mat resampled_map(saliency_map_shape_, CV_32FC1);
      resize(normalized_map, resampled_map, resampled_map.size(),
             0, 0, INTER_NEAREST);
      conspicuity += resampled_map;
    }
    else conspicuity += normalized_map;
  }
  conspicuity /= static_cast<float>(maps.size());
}

// generates the feature maps based on center-surround difference model
void SaliencyMap::GenerateFeatureMaps(const ScaleSpace& pyramid,
                                      std::vector<Mat>& feature_maps)
{
  unsigned int center_level=0;
  unsigned int surround_delta=0;

  const int border = std::max(saliency_map_shape_.width,
                     saliency_map_shape_.height)/20;

  for (center_level = center_level_min_;
       center_level <= center_level_max_;
       ++center_level) {
    for (surround_delta = surround_delta_min_;
         surround_delta <= surround_delta_max_;
         ++surround_delta) {
      Mat feature_map = pyramid.Difference(center_level,
                                           center_level + surround_delta);
//      feature_map = abs(feature_map);
      AttenuateBorders(border, feature_map);
      feature_maps.push_back(feature_map);
    }
  }
}

// generates the feature maps based on excitation-inhibition model
void SaliencyMap::GenerateFeatureMapsEI(const ScaleSpace& pyramid_excitation,
                                        const ScaleSpace& pyramid_inhibition,
                                        std::vector<Mat>& feature_maps)
{
  unsigned int center_level=0;
  unsigned int surround_delta=0;

  const int border = std::max(saliency_map_shape_.width,
                     saliency_map_shape_.height)/20;

  for (center_level = center_level_min_;
       center_level <= center_level_max_;
       ++center_level) {
    for (surround_delta = surround_delta_min_;
         surround_delta <= surround_delta_max_;
         ++surround_delta) {
      Mat excitation =
          pyramid_excitation.Difference(center_level,
                                        center_level + surround_delta);
      Mat inhibition =
          pyramid_inhibition.Difference(center_level,
                                        center_level + surround_delta);

      Mat feature_map(excitation.size(), CV_32FC1);
      absdiff(excitation, inhibition, feature_map); // = excitation - inhibition
      // feature_map = abs(feature_map);
      AttenuateBorders(border, feature_map);
      feature_maps.push_back(feature_map);
    }
  }
}

// attenuate borders
void SaliencyMap::AttenuateBorders(const int size, Mat& image)
{
  int border_size = size;

  if (border_size*2 > image.rows) border_size = image.rows/2;
  if (border_size*2 > image.cols) border_size = image.cols/2;
  if (border_size < 1) return;

  // top and bottom
  for (int y=0; y<border_size; ++y) {
    for (int x=0; x<image.cols; ++x) {
      image.at<float>(y,x) *= (y+1.0F)/(border_size+1.0F);
      image.at<float>(image.rows-y-1,x) *= (y+1.0F)/(border_size+1.0F);
    }
  }

  // left and right
  for (int y=0; y<image.rows; ++y) {
    for (int x=0; x<border_size; ++x) {
      image.at<float>(y,x) *= (x+1.0F)/(border_size+1.0F);
      image.at<float>(y,image.cols-x-1) *= (x+1.0F)/(border_size+1.0F);
    }
  }
}

Size SaliencyMap::getGaborSize(const float sigma, const float gamma,
                               const float theta)
{
  assert( sigma > 0.0f || gamma != 0.0f);

  const float sigma_x = sigma;
  const float sigma_y = sigma/gamma;

  float xmax = std::max(fabs(3.0*sigma_x*cos(theta)),
                        fabs(3.0*sigma_y*sin(theta)));
  xmax = ceil(max(1.0F, xmax));

  float ymax = std::max(fabs(3.0*sigma_x*sin(theta)),
                        fabs(3.0*sigma_y*cos(theta)));
  ymax = ceil(max(1.0F, ymax));

  const int radius_x = static_cast<int>(xmax);
  const int radius_y = static_cast<int>(ymax);

  const int kernel_width  = 2*radius_x+1;
  const int kernel_height = 2*radius_y+1;

  return Size(kernel_width, kernel_height);
}

// Early Visual features -> feature channel
void SaliencyMap::GenerateFeatureChannels(const Mat& image,
                                          Mat& intensity_channel,
                                          Mat& red_channel,
                                          Mat& green_channel,
                                          Mat& blue_channel,
                                          Mat& yellow_channel,
                                          Mat& orient_0d_channel,
                                          Mat& orient_45d_channel,
                                          Mat& orient_90d_channel,
                                          Mat& orient_135d_channel)
{
  assert(image.type() == CV_8UC3);

  const int width  = image.cols;
  const int height = image.rows;

  //----------------------------------------------------------------------------
  // 1. intensity channel

  // obtain an intensity image & find the maximam value for RG, BY channel
  float intensity_threshold = 0.0F;
  intensity_channel.create(height, width, CV_32FC1);

  // temporal channel arrays
  std::vector<Mat> bgr_temp(3);
  split(image, bgr_temp);

  for (int y=0; y<height; ++y) {
    for (int x=0; x<width; ++x) {
      const float intensity = (bgr_temp[0].at<unsigned char>(y,x) +
                               bgr_temp[1].at<unsigned char>(y,x) +
                               bgr_temp[2].at<unsigned char>(y,x)) / 3.0F;
      intensity_channel.at<float>(y,x) = intensity;

      // find the max value to set the threshold value later.
      // intensity_threshold points out the maximam value here.
      if (intensity > intensity_threshold) intensity_threshold = intensity;
    }
  }

  //----------------------------------------------------------------------------
  // 2. red/green/blue/yellow channel

  // 2-1. extracts early visual features
  Mat early_feature_red(  height, width, CV_32FC1);
  Mat early_feature_green(height, width, CV_32FC1);
  Mat early_feature_blue( height, width, CV_32FC1);

  // r,g,b channels are normalized by I to decouple hue from intensity
  //
  // - hue variations are not perceivable at very low luminance (and hence are
  //   not salient) normalization is only applied at the locations where I is
  //   larger than 1/10 of its maximum over the entire image
  //   (other locations yield zero r,g and b)
  intensity_threshold /= 10.0F;

  for (int y=0; y<height; ++y) {
    for (int x=0; x<width; ++x) {
      const float intensity_value = intensity_channel.at<float>(y,x);
      if (intensity_value > intensity_threshold) {
        early_feature_blue.at<float>(y,x)  = bgr_temp[0].at<unsigned char>(y,x)/
                                             intensity_value;
        early_feature_green.at<float>(y,x) = bgr_temp[1].at<unsigned char>(y,x)/
                                             intensity_value;
        early_feature_red.at<float>(y,x)   = bgr_temp[2].at<unsigned char>(y,x)/
                                             intensity_value;
      }
      else {
        early_feature_red.at<float>(y,x)   = 0.0F;
        early_feature_green.at<float>(y,x) = 0.0F;
        early_feature_blue.at<float>(y,x)  = 0.0F;
      }
    }
  }

  // 2-2. generates red/green/blue/yellow channels
  red_channel.create(   height, width, CV_32FC1);
  green_channel.create( height, width, CV_32FC1);
  blue_channel.create(  height, width, CV_32FC1);
  yellow_channel.create(height, width, CV_32FC1);

  for (int y=0; y<height; ++y) {
    for (int x=0; x<width; ++x) {
      red_channel.at<float>(y,x)
          = early_feature_red.at<float>(y,x) -
            (early_feature_green.at<float>(y,x) +
             early_feature_blue.at<float>(y,x)) / 2.0F;
      green_channel.at<float>(y,x)
          = early_feature_green.at<float>(y,x) -
            (early_feature_red.at<float>(y,x) +
             early_feature_blue.at<float>(y,x)) / 2.0F;
      blue_channel.at<float>(y,x)
          = early_feature_blue.at<float>(y,x) -
            (early_feature_red.at<float>(y,x) +
             early_feature_green.at<float>(y,x)) / 2.0F;
      yellow_channel.at<float>(y,x)
         = ((early_feature_red.at<float>(y,x) +
             early_feature_green.at<float>(y,x))/2.0F) -
           (fabs(early_feature_red.at<float>(y,x) -
                 early_feature_green.at<float>(y,x))/2.0F) -
           early_feature_blue.at<float>(y,x);

      // truncated to positive value
      if (red_channel.at<float>(y,x) < 0.0F)
        red_channel.at<float>(y,x) = 0.0F;
      if (green_channel.at<float>(y,x) < 0.0F)
        green_channel.at<float>(y,x) = 0.0F;
      if (blue_channel.at<float>(y,x) < 0.0F)
        blue_channel.at<float>(y,x) = 0.0F;
      if (yellow_channel.at<float>(y,x) < 0.0F)
        yellow_channel.at<float>(y,x) = 0.0F;
    }
  }

  // 3. orientation channel
  const float g_div = 6.2F;
  const float g_gamma = 1.0F;
  const float g_lambda = 11.0F*2.0F/g_div;
  const float g_sigma = g_lambda*0.8F;

  /**
   *  Gabor filtering should be performed in 'reflect' mode to avoid the
   *  boundary artifact.
   */
  Mat gabor_0d   = getGaborKernel(getGaborSize(g_sigma, g_gamma, 0),
                                  g_sigma, 0, g_lambda, g_gamma, 0,
                                  CV_32FC1);
  Mat gabor_45d  = getGaborKernel(getGaborSize(g_sigma, g_gamma, CV_PI/4.0F),
                                  g_sigma, CV_PI/4.0F, g_lambda, g_gamma, 0,
                                  CV_32FC1);
  Mat gabor_90d  = getGaborKernel(getGaborSize(g_sigma, g_gamma, CV_PI/2.0F),
                                  g_sigma, CV_PI/2.0F, g_lambda, g_gamma, 0,
                                  CV_32FC1);
  Mat gabor_135d =
      getGaborKernel(getGaborSize(g_sigma, g_gamma, CV_PI*3.0F/4.0F),
                     g_sigma, CV_PI*3.0F/4.0F, g_lambda, g_gamma, 0,
                     CV_32FC1);

  orient_0d_channel.create(height,width,CV_32FC1);
  orient_45d_channel.create(height,width,CV_32FC1);
  orient_90d_channel.create(height,width,CV_32FC1);
  orient_135d_channel.create(height,width,CV_32FC1);

  // extracts the orientation feature channel
  filter2D(intensity_channel, orient_0d_channel, CV_32F, gabor_0d,
           Point(-1,-1), 0, BORDER_REFLECT_101);
  filter2D(intensity_channel, orient_45d_channel, CV_32F, gabor_45d,
           Point(-1,-1), 0, BORDER_REFLECT_101);
  filter2D(intensity_channel, orient_90d_channel, CV_32F, gabor_90d,
           Point(-1,-1), 0, BORDER_REFLECT_101);
  filter2D(intensity_channel, orient_135d_channel, CV_32F, gabor_135d,
           Point(-1,-1), 0, BORDER_REFLECT_101);
}

bool SaliencyMap::Compute(const Mat& image)
{
  // return false if the given image is gray-scale
  if (image.type() != CV_8UC3) return false;

  // 1. Feature Channel Generation
  Mat intensity_channel;
  Mat red_channel;
  Mat green_channel;
  Mat blue_channel;
  Mat yellow_channel;
  Mat orient_0d_channel;
  Mat orient_45d_channel;
  Mat orient_90d_channel;
  Mat orient_135d_channel;

  GenerateFeatureChannels(image, intensity_channel, /** luminance channel */
                          red_channel, green_channel,
                          blue_channel, yellow_channel, /** color channel */
                          orient_0d_channel, orient_45d_channel,
                          orient_90d_channel, orient_135d_channel
                          /* orientation channel*/);

  //----------------------------------------------------------------------------
  // dump the initial feature maps
  if (debug_) {
    imwrite("1_easaliency_intensity.jpg", ConvertMap(intensity_channel));
    imwrite("1_easaliency_red.jpg", ConvertMap(red_channel));
    imwrite("1_easaliency_green.jpg", ConvertMap(green_channel));
    imwrite("1_easaliency_blue.jpg", ConvertMap(blue_channel));
    imwrite("1_easaliency_yellow.jpg", ConvertMap(yellow_channel));
    imwrite("1_easaliency_ori_0d.jpg", orient_0d_channel);
    imwrite("1_easaliency_ori_45d.jpg", orient_45d_channel);
    imwrite("1_easaliency_ori_90d.jpg", orient_90d_channel);
    imwrite("1_easaliency_ori_135d.jpg", orient_135d_channel);
  }
  //----------------------------------------------------------------------------

  //
  // Nine spatial scales are created using dyadic Gaussian pyramids
  // yielding progressively pass filter and subsample the input image
  //
  ScaleSpace intensity_pyramid(pyramid_level_, intensity_channel);
  ScaleSpace red_pyramid(pyramid_level_, red_channel);
  ScaleSpace green_pyramid(pyramid_level_, green_channel);
  ScaleSpace blue_pyramid(pyramid_level_, blue_channel);
  ScaleSpace yellow_pyramid(pyramid_level_, yellow_channel);
  ScaleSpace orient_0d_pyramid(pyramid_level_, orient_0d_channel);
  ScaleSpace orient_45d_pyramid(pyramid_level_, orient_45d_channel);
  ScaleSpace orient_90d_pyramid(pyramid_level_, orient_90d_channel);
  ScaleSpace orient_135d_pyramid(pyramid_level_, orient_135d_channel);

  if (debug_) {
    ImgDump intensity_pyramid_dump("2_esaliency_intensity_pyramid_");
    ImgDump red_pyramid_dump("2_esaliency_red_pyramid_");
    ImgDump green_pyramid_dump("2_esaliency_green_pyramid_");
    ImgDump blue_pyramid_dump("2_esaliency_blue_pyramid_");
    ImgDump yellow_pyramid_dump("2_esaliency_yellow_pyramid_");
    ImgDump ori0_pyramid_dump("2_esaliency_ori0d_pyramid_");
    ImgDump ori45_pyramid_dump("2_esaliency_ori45d_pyramid_");
    ImgDump ori90_pyramid_dump("2_esaliency_ori90d_pyramid_");
    ImgDump ori135_pyramid_dump("2_esaliency_ori135d_pyramid_");

    intensity_pyramid_dump.Export(intensity_pyramid.get_pyramid(), true);
    red_pyramid_dump.Export(red_pyramid.get_pyramid(), true);
    green_pyramid_dump.Export(green_pyramid.get_pyramid(), true);
    blue_pyramid_dump.Export(blue_pyramid.get_pyramid(), true);
    yellow_pyramid_dump.Export(yellow_pyramid.get_pyramid(), true);
    ori0_pyramid_dump.Export(orient_0d_pyramid.get_pyramid(), true);
    ori45_pyramid_dump.Export(orient_45d_pyramid.get_pyramid(), false);
    ori90_pyramid_dump.Export(orient_90d_pyramid.get_pyramid(), false);
    ori135_pyramid_dump.Export(orient_135d_pyramid.get_pyramid(), false);
  }

  // saliency map size
  saliency_map_shape_ = intensity_pyramid.get_size(saliency_level_);
  // std::cout << "scale space generation done" << std::endl;

  //----------------------------------------------------------------------------
  // feature map generation
  //
  // center-surround difference between a "center" fine scale and a "surround"
  // coarser scale yields the feature maps
  //
  // feature map (RG, BY) : color channel, "color double-opponent" system
  //
  // In the center of the receptive field, neurons are exicted by one color
  // (e.g., red) and inhibited by another (e.g., green), while the converse is
  // true in the surround. Such spatial and chromatic opponency exists for the
  // red/green, blue/yellow pairs in human primary visual cortex.
  std::vector<Mat> intensity_feature_maps;
  std::vector<Mat> rg_color_feature_maps;
  std::vector<Mat> by_color_feature_maps;
  std::vector<Mat> orient_0d_feature_maps;
  std::vector<Mat> orient_45d_feature_maps;
  std::vector<Mat> orient_90d_feature_maps;
  std::vector<Mat> orient_135d_feature_maps;

  GenerateFeatureMaps(intensity_pyramid, intensity_feature_maps);
  GenerateFeatureMapsEI(red_pyramid, green_pyramid, rg_color_feature_maps);
  GenerateFeatureMapsEI(blue_pyramid, yellow_pyramid, by_color_feature_maps);
  GenerateFeatureMaps(orient_0d_pyramid, orient_0d_feature_maps);
  GenerateFeatureMaps(orient_45d_pyramid, orient_45d_feature_maps);
  GenerateFeatureMaps(orient_90d_pyramid, orient_90d_feature_maps);
  GenerateFeatureMaps(orient_135d_pyramid, orient_135d_feature_maps);

  if (debug_) {
    ImgDump intensity_feature_dump("3_esaliency_intensity_feature_");
    ImgDump rg_feature_dump("3_esaliency_rg_feature_");
    ImgDump by_feature_dump("3_esaliency_by_feature_");
    ImgDump ori0_feature_dump("3_esaliency_ori0d_feature_");
    ImgDump ori45_feature_dump("3_esaliency_ori45d_feature_");
    ImgDump ori90_feature_dump("3_esaliency_ori90d_feature_");
    ImgDump ori135_feature_dump("3_esaliency_ori135d_feature_");

    intensity_feature_dump.Export(intensity_feature_maps, true);
    rg_feature_dump.Export(rg_color_feature_maps, true);
    by_feature_dump.Export(by_color_feature_maps, true);
    ori0_feature_dump.Export(orient_0d_feature_maps, true);
    ori45_feature_dump.Export(orient_45d_feature_maps, false);
    ori90_feature_dump.Export(orient_90d_feature_maps, false);
    ori135_feature_dump.Export(orient_135d_feature_maps, false);
  }
  // std::cout << "feature map generation done" << std::endl;

  //----------------------------------------------------------------------------
  // Conspicuity Maps Generation : Across-scale combination & normalization
  //
  // Combining multi-scale feature maps, from different visual modalities with
  // unrelated dynamic ranges into a unique saliency map
  //
  // There are 4 ways to implement the conspicuity map
  //
  // 1. Simple normalized summation
  // 2. Linear combination with learned weights
  // 3. Global non-linear normalization followed by summation
  // 4. Local non-linear competition between salient locations
  //
  // ABOUT Normalization operator N()
  //    1) Normalizing the values in the map to a fixed range [0..M], in order
  //       to eliminate modality-dependent amplitude differences
  //    2) finding the location of the map's global maximum M and computing
  //       the average ~m of all its other local maxima
  //    3) globally multiplying the map by (M-~m)^2
  //
  // Why Normalization?
  //    Different modalities contribute independently to the saliency map.
  //

  //----------------------------------------------------------------------------
  // Intensity conspicuity map [0 .. 1]
  Mat intensity_cmap;
  GenerateConspicuity(intensity_feature_maps, intensity_cmap);
  conspicuity_map_intensity_ = normalizer_->Compute(intensity_cmap);

  //----------------------------------------------------------------------------
  // Color conspicuity map
  Mat rgcolor_conspicuity;
  GenerateConspicuity(rg_color_feature_maps, rgcolor_conspicuity);

  Mat bycolor_conspicuity;
  GenerateConspicuity(by_color_feature_maps, bycolor_conspicuity);

  Mat color_cmap(saliency_map_shape_, CV_32FC1);
  color_cmap = (rgcolor_conspicuity + bycolor_conspicuity) / 2.0F;
  conspicuity_map_color_ = normalizer_->Compute(color_cmap);

  //----------------------------------------------------------------------------
  // Orientation conspicuity map
  std::vector<Mat> orient_alld_conspicuity;

  Mat orient_0d_conspicuity;
  GenerateConspicuity(orient_0d_feature_maps, orient_0d_conspicuity);
  orient_alld_conspicuity.push_back(orient_0d_conspicuity);

  Mat orient_45d_conspicuity;
  GenerateConspicuity(orient_45d_feature_maps, orient_45d_conspicuity);
  orient_alld_conspicuity.push_back(orient_45d_conspicuity);

  Mat orient_90d_conspicuity;
  GenerateConspicuity(orient_90d_feature_maps, orient_90d_conspicuity);
  orient_alld_conspicuity.push_back(orient_90d_conspicuity);

  Mat orient_135d_conspicuity;
  GenerateConspicuity(orient_135d_feature_maps, orient_135d_conspicuity);
  orient_alld_conspicuity.push_back(orient_135d_conspicuity);

  Mat orientation_cmap;
  GenerateConspicuity(orient_alld_conspicuity, orientation_cmap);
  conspicuity_map_orientation_ = normalizer_->Compute(orientation_cmap);

  if (debug_) {
    ImgDump intensity_conspicuity_dump("4_esaliency_intensity_cmap_");
    ImgDump rg_conspicuity_dump("4_esaliency_rg_cmap_");
    ImgDump by_conspicuity_dump("4_esaliency_by_cmap_");
    ImgDump ori0_conspicuity_dump("4_esaliency_ori0d_cmap_");
    ImgDump ori45_conspicuity_dump("4_esaliency_ori45d_cmap_");
    ImgDump ori90_conspicuity_dump("4_esaliency_ori90d_cmap_");
    ImgDump ori135_conspicuity_dump("4_esaliency_ori135d_cmap_");
    ImgDump int_conspicuity_dump("5_esaliency_intensity_conspicuity_");
    ImgDump col_conspicuity_dump("5_esaliency_color_conspicuity_");
    ImgDump ori_conspicuity_dump("5_esaliency_orientation_conspicuity_");

    intensity_conspicuity_dump.Export(intensity_cmap, true);
    rg_conspicuity_dump.Export(rgcolor_conspicuity, true);
    by_conspicuity_dump.Export(bycolor_conspicuity, true);
    ori0_conspicuity_dump.Export(orient_0d_conspicuity, false);
    ori45_conspicuity_dump.Export(orient_45d_conspicuity, false);
    ori90_conspicuity_dump.Export(orient_90d_conspicuity, false);
    ori135_conspicuity_dump.Export(orient_135d_conspicuity, false);

    int_conspicuity_dump.Export(conspicuity_map_intensity_, true);
    col_conspicuity_dump.Export(conspicuity_map_color_, true);
    ori_conspicuity_dump.Export(conspicuity_map_orientation_, true);
  }

  //----------------------------------------------------------------------------
  // Saliency Map Generation
  saliency_map_.create(saliency_map_shape_, CV_32FC1);
  saliency_map_ = (conspicuity_map_intensity_ +
                   conspicuity_map_color_ +
                   conspicuity_map_orientation_) / 3.0F;

  return true;
}

}