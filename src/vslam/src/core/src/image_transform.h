#ifndef VSLAM_CORE_IMAGE_TRANSFORM_H__
#define VSLAM_CORE_IMAGE_TRANSFORM_H__
#include <Eigen/Dense>
#include "types.h"
namespace pd::vslam {

  template < typename Derived, typename Operation >
  void forEachPixel(const Eigen::Matrix < Derived, -1, -1 > & image, Operation op)
  {
    //give option to parallelize?
    for (int v = 0; v < image.rows(); v++) {
      for (int u = 0; u < image.cols(); u++) {
        op(u, v, image(v, u));
      }
    }
  }
}

#endif
