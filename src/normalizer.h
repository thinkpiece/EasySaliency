/**
 *  @file   normalizer.h
 *  @author Junyoung Park (thinkpiece@yahoo.com)
 *  @brief  normalize various maps before combining them
 *
 *  The normalization ensures that the different modalities resulting from the
 *  previous processing steps are made comparable. Furthermore, it promotes maps
 *  with few dominating peaks, and suppress maps with many similar peaks.
 *  For more information, please see the paper below.
 *
 *  L. Itti and C. Koch, “A saliency-based search mechanism for overt and
 *  covert shifts of visual attention,” Vision Research, vol. 40, no. 10, pp.
 *  1489–1506, Jun. 2000.
 *
 */
#ifndef __EASY_SALIENCY_NORMALIZER_H__
#define __EASY_SALIENCY_NORMALIZER_H__

#include <opencv2/core/core.hpp>

namespace cv
{

/** Normalize the given image
 *
 *  Compute() globally promotes maps in which a small number of strong
 *  peaks of activity (conspicuity location) is present, while globally
 *  suppressing map which contains numerous comparable peak response.
 */
class CV_EXPORTS Normalizer
{
 public:
  static Normalizer* Create() { return 0; }
  virtual Mat Compute(const Mat& image) = 0;

 protected:
  Normalizer() { }
  Normalizer(const Normalizer& ref) { }
};

/** Normalization method based on local maximum
 * @note
 *  1. Normalizing the values in the map to a fixed range [0..M], in order to
 *     eliminate modality-dependent amplitude differences
 *  2. finding the location of the map's global maximum M and
 *  3. computing the average ~m of all its other local maxima
 *  4. globally multiplying the map by (M-~m)^2
 */
class CV_EXPORTS LocalMaxNormalizer : public Normalizer
{
 public:
  static LocalMaxNormalizer* Create()
  {
    if (pinst_ == 0) pinst_ = new LocalMaxNormalizer;
    return pinst_;
  }
  Mat Compute(const Mat& image);

 private:
  LocalMaxNormalizer() { }
  LocalMaxNormalizer(const LocalMaxNormalizer& ref) { }
  static LocalMaxNormalizer* pinst_;
};

/** Normalization method based on */
class CV_EXPORTS IterativeNormalizer : public Normalizer
{
 public:
  static IterativeNormalizer* Create()
  {
    if (pinst_ == 0) pinst_ = new IterativeNormalizer;
    return pinst_;
  }
  Mat Compute(const Mat& image);

  void set_inhibition_constant(float value) { inhibition_constant_ = value;}
  void set_excitation_sigma(float value)    { excitation_sigma_ = value; }
  void set_inhibition_sigma(float value)    { inhibition_sigma_ = value; }
  void set_excitation_coef(float value)     { excitation_coef_ = value; }
  void set_inhibition_coef(float value)     { inhibition_coef_ = value; }
  void set_iteration(float value)           { iteration_ = value; }

 private:
  IterativeNormalizer();
  IterativeNormalizer(const IterativeNormalizer& ref) { }

  Mat PreserveEnergy(const Mat& kernel, const Mat& src);

  /** @name Some parameters for Iterative Normalization Method */
  ///@{
  float inhibition_constant_;
  float excitation_sigma_;
  float inhibition_sigma_;
  float excitation_coef_;
  float inhibition_coef_;
  float iteration_;
  ///@}
  static IterativeNormalizer* pinst_;
};

} // end namespace es


#endif