//
// Created by phil on 30.06.21.
//

#ifndef VSLAM_POINT_H
#define VSLAM_POINT_H

#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace pd::vslam {
  class Feature2D;

  class Point3D {
public:
    using ShPtr = std::shared_ptr < Point3D >;
    using ConstShPtr = std::shared_ptr < Point3D >;
    Point3D(const Eigen::Vector3d & position, std::shared_ptr < Feature2D > ft);
    Point3D(
      const Eigen::Vector3d & position,
      const std::vector < std::shared_ptr < Feature2D >> &features);
    void addFeature(std::shared_ptr < Feature2D > ft);
    void removeFeatures();
    void removeFeature(std::shared_ptr < Feature2D > f);
    void remove();

    const Eigen::Vector3d & position() const {return _position;}
    Eigen::Vector3d & position() {return _position;}

    std::vector < std::shared_ptr < Feature2D >> features() {
      return _features;
    }
    std::vector < std::shared_ptr <
    const Feature2D >>
    features() const {return std::vector < std::shared_ptr <
             const Feature2D >> (_features.begin(), _features.end());}
    std::uint64_t id() const {return _id;}

private:
    const std::uint64_t _id;
    Eigen::Vector3d _position;
    std::vector < std::shared_ptr < Feature2D >> _features;
    static std::uint64_t _idCtr;
  };

}
#endif //VSLAM_POINT_H
