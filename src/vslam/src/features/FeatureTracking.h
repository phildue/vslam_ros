#pragma once
#include "core/Feature2D.h"
#include "core/Frame.h"
#include "core/macros.h"
#include "descriptor_matching/overlays.h"
#include "keypoint_selection.h"
#include "utils/log.h"
namespace vslam {

class FeatureTracking {
public:
  typedef std::shared_ptr<FeatureTracking> ShPtr;
  typedef std::unique_ptr<FeatureTracking> UnPtr;

  static constexpr char LOG_NAME[] = "features";

  struct Correspondence {
    Feature2D::ShPtr ft0;
    Vec3d p3dw;
    Vec2d uv1;
    double score;
  };

  struct FullScore {
    double operator()(const Correspondence &UNUSED(c)) const { return 1.0; }
  };

  FeatureTracking(float threshold, float gridSize) :
      _threshold(threshold),
      _gridSize(gridSize) {
    log::create(LOG_NAME);
  }

  template <typename Score = FullScore> Point3D::VecShPtr track(Frame::ShPtr f0, Frame::ShPtr f1, const Score &score = FullScore()) const {
    return track(f0->features(), f1, score);
  }
  template <typename Score = FullScore>
  Point3D::VecShPtr track(Frame::VecShPtr fs0, Frame::ShPtr f1, const Score &score = FullScore()) const {
    Feature2D::VecShPtr candidates0;
    for (auto f0 : fs0) {
      auto features = f0->features();
      candidates0.insert(candidates0.end(), features.begin(), features.end());
    }
    return track(candidates0, f1, score);
  }
  template <typename Score = FullScore>
  Point3D::VecShPtr track(Feature2D::VecShPtr candidates0, Frame::ShPtr f1, const Score &score = FullScore()) const {

    std::vector<Correspondence> correspondences(candidates0.size());

    std::transform(candidates0.begin(), candidates0.end(), correspondences.begin(), [&](auto ft) {
      const Vec3d p3dw = ft->frame()->p3dWorld(ft->v(), ft->u());
      const Vec2d uv1 = f1->world2image(p3dw);
      Correspondence c{ft, p3dw, uv1, 0.};
      c.score = score(c);
      return c;
    });
    correspondences.erase(
      std::remove_if(correspondences.begin(), correspondences.end(), [&](const auto &c) { return score(c) < _threshold; }),
      correspondences.end());

    // TODO do this for each level separately? NO
    correspondences = keypoint::subsampling::grid(
      correspondences,
      f1->height(),
      f1->width(),
      _gridSize,
      [&](auto c0, auto c1) { return c0.score > c1.score; },
      [](auto c) { return c.uv1; });

    Feature2D::VecShPtr features1(correspondences.size());
    // TODO how to provide flexibility on the feature response?
    std::transform(correspondences.begin(), correspondences.end(), features1.begin(), [&](const auto &c) {
      const float scale = 1.0f / std::pow(2.0f, c.ft0->level());
      const cv::Vec2f dIuv = cv::Mat(f1->dI(c.ft0->level())).at<cv::Vec2f>(c.uv1(1) * scale, c.uv1(0) * scale);
      const Vec2d response{dIuv[0], dIuv[1]};
      return std::make_unique<Feature2D>(c.uv1, f1, c.ft0->level(), response.norm());
    });
    f1->addFeatures(features1);

    Point3D::VecShPtr points(correspondences.size());
    std::transform(correspondences.begin(), correspondences.end(), features1.begin(), points.begin(), [&](const auto &c, auto ft1) {
      c.ft0->point() = c.ft0->point() ? c.ft0->point() : std::make_shared<Point3D>(c.p3dw, Feature2D::VecShPtr{c.ft0});
      c.ft0->point()->addFeature(ft1);
      ft1->point() = c.ft0->point();
      CLOG(DEBUG, LOG_NAME) << format(
        "Tracking point {} at {} in frame {} with {} observations.",
        c.ft0->point()->id(),
        c.ft0->point()->position().transpose(),
        f1->id(),
        c.ft0->point()->features().size());

      return c.ft0->point();
    });
    CLOG(INFO, LOG_NAME) << format("Propagated features among {} to frame {}: {}", candidates0.size(), f1->id(), f1->features().size());
    log::append("TrackedFeatures", overlay::Features(f1, _gridSize));

    return points;
  }

  const float &gridSize() const { return _gridSize; }

private:
  const float _threshold;
  const float _gridSize;
};

}  // namespace vslam
