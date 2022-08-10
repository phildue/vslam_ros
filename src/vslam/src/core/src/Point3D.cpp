//
// Created by phil on 30.06.21.
//


#include "Point3D.h"
#include "Feature2D.h"
#include "Frame.h"
#include "Exceptions.h"
#include "Frame.h"
namespace pd::vslam
{

std::uint64_t Point3D::_idCtr = 0U;

Point3D::Point3D(const Eigen::Vector3d & position, std::shared_ptr<Feature2D> ft)
: _id(_idCtr++),
  _position(position)
{
  addFeature(ft);
}

Point3D::Point3D(
  const Eigen::Vector3d & position,
  const std::vector<std::shared_ptr<Feature2D>> & features)
: _id(_idCtr++),
  _position(position)
{
  for (const auto & ft : features) {
    addFeature(ft);
  }
}

void Point3D::addFeature(std::shared_ptr<Feature2D> ft)
{
  _features.push_back(ft);
}

void Point3D::removeFeatures()
{
  for (const auto & ft : _features) {
    ft->point() = nullptr;
    ft->frame()->removeFeature(ft);
  }
  _features.clear();
}

void Point3D::removeFeature(std::shared_ptr<Feature2D> ft)
{
  auto it = std::find(_features.begin(), _features.end(), ft);

  if (it == _features.end()) {
    throw pd::Exception(
            "Did not find feature: [" + std::to_string(
              ft->id()) + " ] in point: [" + std::to_string(_id) + "]");
  }
  ft->point() = nullptr;
  _features.erase(it);

  if (_features.size() < 2) {
    remove();
  }

}
void Point3D::remove()
{
  for (const auto & ftp : _features) {
    ftp->point() = nullptr;
    ftp->frame()->removeFeature(ftp);
  }
}


}
