/**
 *  @file   saliency.h
 *  @author Junyoung Park (thinkpiece@yahoo.com)
 *  @brief  Saliency-based Visual Attention Implementation
 *
 *  Simple implementation for saliency-based visual attention model, which was
 *  proposed by Itti. See below for details:
 *
 *  H. Greenspan, et al., "Overcomplete steerable pyramid filters and rotation
 *  invariance," Proc. IEEE CVPR, 1994.
 *
 */
#ifndef __EASY_SALIENCY_H__
#define __EASY_SALIENCY_H__

#include "scalespace.h"
#include "normalizer.h"

#include <vector>
#include <opencv2/core/core.hpp>

namespace cv
{

class SaliencyMap
{
 public:

  /** there are two normalization method supported */
  enum NormalizationMethod
  {
    kLocalMaxNormalize,
    kIterativeNormalize
  };

  /** default constructor */
  SaliencyMap(NormalizationMethod norm_method=kIterativeNormalize,
              bool ifdebug=false);
  ~SaliencyMap();

  bool Compute(const Mat& image);

  /** returns the saliency map */
  Mat get_saliency_map() const;

  /** returns the conspicuity map for intensity */
  Mat get_conspicuity_map_intensity() const;

  /** returns the conspicuity map for color */
  Mat get_conspicuity_map_color() const;

  /** returns the conspicuity map for orientation */
  Mat get_conspicuity_map_orientation() const;

  /** @name Setting Parameters */
  ///@{
  /** set the center level */
  void set_center_pyramid_level(unsigned int min, unsigned int max);

  /** set the surround level */
  void set_surround_pyramid_level(unsigned int min, unsigned int max);

  /** set the saliency map level */
  void set_saliency_map_level(unsigned int level);
  ///@}

protected:
  /** generates feature channels
   *
   * @note
   *  In this implementation, a total of 9 channels are used as features.
   *  Intensity, Red color, Green color, Blue color, Yellow color,
   *  Orientation with 0 degree, 45 degree, 90 degree, 135 degree.
   */
  void GenerateFeatureChannels(const Mat& image,
                               Mat& intensity_channel,
                               Mat& red_channel,
                               Mat& green_channel,
                               Mat& blue_channel,
                               Mat& yellow_channel,
                               Mat& orient_0_channel,
                               Mat& orient_45_channel,
                               Mat& orient_90_channel,
                               Mat& orient_135_channel);

/** generates the feature maps based on center-surround difference model
 *
 * @note
 *   Center-surround difference between a "center" fine scale and a "surround"
 *   coarser scale yields the feature maps.
 */
  void GenerateFeatureMaps(const ScaleSpace& pyramid,
                           std::vector<Mat>& feature_maps);

/** generates the feature maps based on excitation-inhibition model
 *
 * @note
 *   For the color feature maps (Red-Green, and Blue-Yellow), a theory
 *   a.k.a. "color double-opponent" system is adopted in this implementation.
 *   According to the theory, in the center of the receptive field, neurons
 *   are exicted by one color (e.g., red) and inhibited by another (e.g., green),
 *   while the converse is true in the surround. Such spatial and chromatic
 *   opponency exists for the red/green, blue/yellow pairs in human primary
 *   visual cortex.
 */
  void GenerateFeatureMapsEI(const ScaleSpace& pyramid_excitation,
                             const ScaleSpace& pyramid_inhibition,
                             std::vector<Mat>& feature_maps);

  void GenerateConspicuity(const std::vector<Mat>& maps,
                           Mat& conspicuity);

  void AttenuateBorders(const int size, Mat& image);
  Size getGaborSize(const float sigma, const float gamma,
                    const float theta);

  // void testMap(const Mat& testMap, const char *name);
  Mat ConvertMap(const Mat& image) const;

private:
  Size saliency_map_shape_;

  unsigned int pyramid_level_;
  unsigned int saliency_level_;
  unsigned int center_level_min_;
  unsigned int center_level_max_;
  unsigned int surround_delta_min_;
  unsigned int surround_delta_max_;

  Normalizer *normalizer_;

  Mat conspicuity_map_intensity_;
  Mat conspicuity_map_color_;
  Mat conspicuity_map_orientation_;
  Mat saliency_map_;

  // for debug mode
  bool debug_;
};

}

#endif
