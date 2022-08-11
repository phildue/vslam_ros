#include "MapOptimization.h"
#include "utils/utils.h"
#define LOG_BA(level) CLOG(level, "mapping")

namespace pd::vslam::mapping
{
MapOptimization::MapOptimization()
{
  Log::get("mapping", ODOMETRY_CFG_DIR "/log/mapping.conf");

}
void MapOptimization::optimize(
  const std::vector<FrameRgbd::ShPtr> & frames,
  const std::vector<Point3D::ShPtr> & points) const
{
  BundleAdjustment::UnPtr ba = std::make_unique<BundleAdjustment>();
  for (const auto & p : points) {
    ba->setPoint(p->id(), p->position());
  }
  for (const auto & f : frames) {
    ba->setFrame(f->id(), f->pose().pose(), f->camera()->K());
    for (const auto & ft : f->features()) {
      ba->setObservation(ft->point()->id(), f->id(), ft->position());
    }
  }
  ba->optimize();
  for (const auto & p : points) {
    p->position() = ba->getPoint(p->id());
  }
  for (const auto & f : frames) {
    f->set(PoseWithCovariance(ba->getPose(f->id()), MatXd::Identity(6, 6)));
  }
}
}
