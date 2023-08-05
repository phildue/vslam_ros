

#include <gtest/gtest.h>
using namespace testing;

#include <opencv2/highgui.hpp>
#include <thread>

#include "descriptor_matching/overlays.h"
#include "vslam/core.h"
#include "vslam/direct.h"
#include "vslam/evaluation.h"
#include "vslam/features.h"
#include "vslam/motion_model.h"
#include "vslam/utils.h"
using namespace vslam;

/*
Attempt:

Result:

Explanation:
- For direct methods it's ideal to have very closely resembling frames, as appearance changes can have quite high impact
- Only if there are frames which are not suitable for tracking at all, e.g. due to motion blur it can help to use another reference frame
- The uncertainty measure based on fishers criterion makes some assumptions on the underlying functions which are not met in our case

*/

class ScoreAlignment {

  const float _gridSize;
  const Frame::ConstShPtr _f;
  std::vector<std::vector<double>> _grid;

public:
  typedef std::shared_ptr<ScoreAlignment> ShPtr;
  ScoreAlignment(Frame::ConstShPtr f, float gridSize) :
      _gridSize(gridSize),
      _f(f),
      _grid(std::vector<std::vector<double>>(f->height() / gridSize, std::vector<double>(f->width() / gridSize, 1.))) {}
  ScoreAlignment(Frame::ConstShPtr f, AlignmentRgbd::Results::ConstShPtr r, float gridSize) :
      _gridSize(gridSize),
      _f(f),
      _grid(std::vector<std::vector<double>>(f->height() / gridSize, std::vector<double>(f->width() / gridSize, 1.))) {

    double meanWeight = std::accumulate(r->constraints[0].begin(), r->constraints[0].end(), 0.0, [](auto s, auto c) {
      s += c->weight.norm();
      return s;
    });
    meanWeight /= r->constraints[0].size();
    std::vector<AlignmentRgbd::Constraint::ConstShPtr> constraints(r->constraints[0].begin(), r->constraints[0].end());
    std::sort(constraints.begin(), constraints.end(), [f](auto c0, auto c1) {
      return f->depth(c0->uv1(1), c0->uv1(0)) < f->depth(c1->uv1(1), c1->uv1(0));
    });
    std::for_each(constraints.rbegin(), constraints.rend(), [&](auto c) {
      _grid[c->uv1(1) / gridSize][c->uv1(0) / gridSize] = c->valid ? c->weight.norm() / meanWeight : 0.;
    });
    const int nOccupied = std::accumulate(_grid.begin(), _grid.end(), 0, [meanWeight](auto s, auto row) {
      s += std::accumulate(row.begin(), row.end(), 0.0, [meanWeight](auto vv, auto v) { return v > 1.0 ? vv + 1 : vv; });
      return s;
    });
    const int nCols = f->width(0) / _gridSize;
    const int nRows = f->height(0) / _gridSize;
    CLOG(INFO, "features") << format("Masked {} outliers in {}x{} cells at level {}.", nOccupied, nRows, nCols, 0);
  }
  bool operator()(const Vec2d &uv, Frame::ConstShPtr UNUSED(f), int level) const {
    return _grid[uv(1) * std::pow(2.0, level) / _gridSize][uv(0) * std::pow(2.0, level) / _gridSize] < 0.99;
  }
  double operator()(const FeatureTracking::Correspondence &c) const {
    if (!c.uv1.allFinite() || !_f->withinImage(c.uv1)) {
      return 0.;
    }
    return _grid[c.uv1(1) / _gridSize][c.uv1(0) / _gridSize];
  }
};

class Map {
public:
  Map(int maxKeyFrames) :
      _maxKeyFrames(maxKeyFrames) {}

  void addFrame(Frame::ShPtr f) {
    _childFrames[_keyframes.back()->id()].push_back(f);
    for (const auto &ft : f->featuresWithPoints()) {
      _points[ft->point()->id()] = ft->point();
    }
  }

  void addKeyFrame(Frame::ShPtr f) {

    _keyframes.push_back(f);
    for (const auto &ft : f->featuresWithPoints()) {
      _points[ft->point()->id()] = ft->point();
    }
    if (_keyframes.size() > _maxKeyFrames) {
      _keyframes.front()->removeFeatures();
      for (const auto &cf : _childFrames[_keyframes.front()->id()]) {
        cf->removeFeatures();
      }
      _childFrames.erase(_keyframes.front()->id());
      _keyframes.erase(_keyframes.begin());
      std::erase_if(_points, [](auto p) { return p.second->features().size() < 2; });
    }

    _childFrames[f->id()] = std::vector<Frame::ShPtr>();

    const float nObs = std::accumulate(
      _points.begin(), _points.end(), 0.0, [&](auto s, auto p) { return s + p.second->features().size() / (float)_points.size(); });
    const float nFrames = std::accumulate(
      _childFrames.begin(), _childFrames.end(), _childFrames.size(), [](auto s, auto cfs) { return s + cfs.second.size(); });

    LOG(INFO) << format("Tracking {} points which are on average observed in {:.2f} out of {} frames", _points.size(), nObs, nFrames);

    log::append("KeyFrames", [&]() { return overlay::frames({_keyframes.begin(), _keyframes.end()}, 3, 3, 240, 320); });
    log::append("Correspondences", overlay::CorrespondingPoints({_keyframes.begin(), _keyframes.end()}, 3, 3, 480, 640));
  }

  Feature2D::VecShPtr selectVisibleFeatures(Frame::ConstShPtr f) {
    Feature2D::VecShPtr features;
    std::map<size_t, Feature2D::VecShPtr> featuresAll;
    for (const auto &kf_ : keyframes()) {
      for (const auto &ft : kf_->features()) {
        auto uv1 = f->world2image(ft->frame()->p3dWorld(ft->v(), ft->u()));
        if (!uv1.allFinite() || !f->withinImage(uv1)) {
          continue;
        }
        if (ft->point()) {
          featuresAll[ft->point()->id()].push_back(ft);
        } else {
          features.push_back(ft);
        }
      }
    }
    for (const auto &p_ft : featuresAll) {
      features.push_back(p_ft.second.back());
    }
    return features;
  }
  const Point3D::MapShPtr &points() const { return _points; }
  Frame::ShPtr keyframe() { return _keyframes.back(); }
  Frame::VecShPtr &keyframes() { return _keyframes; }
  Frame::VecConstShPtr keyframes() const { return {_keyframes.begin(), _keyframes.end()}; }
  const std::map<size_t, Frame::VecShPtr> &childFrames() { return _childFrames; }

  void align() {
    return;
    // TODO
    auto poseGraph = std::make_shared<AlignmentRgbPoseGraph>(2, 30, 20.0);
    std::map<size_t, std::vector<Pose>> relativePoses;
    for (const auto &kf : _keyframes) {
      for (const auto &cf : _childFrames[kf->id()]) {
        relativePoses[kf->id()].push_back(cf->pose() * kf->pose().inverse());
      }
    }
    poseGraph->align(_keyframes, {_keyframes.front()});
    for (const auto &kf : _keyframes) {
      for (size_t i = 0; i < relativePoses[kf->id()].size(); i++) {
        _childFrames[kf->id()][i]->pose() = relativePoses[kf->id()][i] * kf->pose();
      }
    }
  }

private:
  const size_t _maxKeyFrames = 7;
  Frame::VecShPtr _keyframes;
  std::map<size_t, Frame::VecShPtr> _childFrames;
  Point3D::MapShPtr _points;
};

namespace vslam::overlay {
struct MapReprojection {

  const std::shared_ptr<const Map> map;
  const Frame::ConstShPtr f;
  cv::Mat operator()() const { return draw(); }
  cv::Mat draw() const {
    Feature2D::VecConstShPtr features;
    std::map<size_t, Feature2D::VecConstShPtr> featuresAll;
    for (const auto &kf_ : map->keyframes()) {
      for (const auto &ft : kf_->features()) {
        if (ft->point()) {
          featuresAll[ft->point()->id()].push_back(ft);
        } else {
          features.push_back(ft);
        }
      }
    }
    for (const auto &p_ft : featuresAll) {
      features.push_back(p_ft.second.back());
    }
    cv::Mat mat = visualizeFrame(f);
    int i = 0;
    for (auto ft : features) {
      auto uv1 = f->world2image(ft->frame()->p3dWorld(ft->v(), ft->u()));
      if (uv1.allFinite() && f->withinImage(uv1)) {
        cv::Point center(uv1(0), uv1(1));
        const double radius = 1 * std::pow(2.0, ft->level());
        cv::circle(mat, center, radius, cv::Scalar(0, 255, 0), 2);
        if (i++ % 100 == 0) {
          cv::putText(
            mat, format("{}", ft->frame()->id()), cv::Point(uv1(0) + 5, uv1(1)), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));
        }
      }
    }
    return mat;
  }
};

}  // namespace vslam::overlay

int main(int argc, char **argv) {
  const std::string filename = argv[0];
  const std::string experimentId = filename.substr(filename.find_last_of("/") + 1);
  const std::vector<std::string> sequences = evaluation::tum::sequencesTraining();
  random::init(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  std::string sequenceId = argc > 1 ? argv[1] : "";
  if (sequenceId.empty() || sequenceId == "random") {
    sequenceId = sequences[random::U(0, sequences.size() - 1)];
  }
  auto dl = std::make_unique<evaluation::tum::DataLoader>("/mnt/dataset/tum_rgbd/", sequenceId);

  Trajectory::ShPtr trajGt = evaluation::tum::loadTrajectory(dl->pathGt());
  const std::string outPath = format("{}/algorithm_results/{}", dl->sequencePath(), experimentId);
  const std::string trajectoryAlgoPath = format("{}/{}-algo.txt", outPath, dl->sequenceId());
  const size_t nFrames = std::min(-1UL, dl->nFrames());
  const size_t fNo0 = nFrames >= dl->nFrames() ? 0UL : random::U(0UL, dl->nFrames() - nFrames);

  const int tRmse = 200;
  std::thread thread;
  log::initialize(outPath, true);
  log::configure(TEST_RESOURCE "/log/");
  log::config("Frame")->show = 1;
  log::config("Features")->show = 1;
  // log::config("TrackedFeatures")->show = 0;
  log::config("PredictedFeatures")->show = -1;
  log::config("AlignedFeatures")->show = -1;
  log::config("AlignedFeaturesMap")->show = 1;
  log::config("MapReprojection")->show = -1;
  log::config("KeyFrames")->show = 1;

  auto directIcp = std::make_shared<AlignmentRgbd>(AlignmentRgbd::defaultParameters());
  auto motionModel = std::make_shared<ConstantVelocityModel>(10.0, INFd, INFd);
  auto featureSelection = std::make_shared<FeatureSelection<FiniteGradient>>(FiniteGradient{5, 0.01, 0.3, 0, 8.0}, 10, 1);
  auto featureTracking = std::make_shared<FeatureTracking>(0.75, 10);
  auto map = std::make_shared<Map>(7);

  Trajectory::ShPtr traj = std::make_shared<Trajectory>();
  const size_t fEnd = dl->timestamps().size();
  Frame::ShPtr kf = dl->loadFrame(fNo0);
  kf->computePyramid(directIcp->nLevels());
  kf->computeDerivatives();
  kf->computePcl();
  featureSelection->select(kf);
  for (const auto &ft : kf->features()) {
    ft->point() = std::make_shared<Point3D>(kf->p3d(ft->v(), ft->u()), ft);
  }
  motionModel->update(kf->pose(), kf->t());
  map->addKeyFrame(kf);
  Frame::ShPtr lf = kf;
  ScoreAlignment::ShPtr lfScore = std::make_shared<ScoreAlignment>(lf, featureSelection->gridSize());
  traj->append(kf->t(), kf->pose().inverse());
  double entropyRef = 0.;
  for (size_t fId = fNo0 + 1; fId < fNo0 + nFrames; fId++) {

    Frame::ShPtr f = dl->loadFrame(fId);
    f->computePyramid(directIcp->nLevels());
    f->computeDerivatives();
    f->computePcl();

    f->pose() = motionModel->predict(f->t());

    log::append("Frame", [&]() { return overlay::frames({kf, lf, f}); });
    ScoreAlignment::ShPtr score = std::make_shared<ScoreAlignment>(f, featureSelection->gridSize());
    f->pose() = directIcp->align(kf, f)->pose;
    const double entropyRatio = std::log(f->pose().cov().determinant()) / entropyRef;
    log::append("MapReprojection", overlay::MapReprojection{map, f});
    Feature2D::VecShPtr features = map->selectVisibleFeatures(f);
    AlignmentRgbd::Results::ConstShPtr results = directIcp->align({features.begin(), features.end()}, f);
    f->pose() = results->pose;
    score = std::make_shared<ScoreAlignment>(f, results, 2 * featureSelection->gridSize());
    featureTracking->track(features, f, *score);
    log::append("AlignedFeaturesMap", overlay::Features(f, 2 * featureSelection->gridSize()));

    print(
      "{}/{}: {} m, {:.3f}°  {:.3f} s |H|={:.3f}\n",
      fId,
      fEnd,
      f->pose().translation().norm(),
      f->pose().totalRotationDegrees(),
      (f->t() - lf->t()) / 1e9,
      entropyRatio);

    if (lf != kf && (entropyRatio < 0.9 || f->features().size() < 500)) {
      print("Keyframe selected.\n");

      log::append("MapReprojection", overlay::MapReprojection(map, lf));

      map->addKeyFrame(lf);
      map->align();
      for (const auto &kf_ : map->keyframes()) {
        traj->append(kf_->t(), kf_->pose());
        for (const auto &cf_ : map->childFrames().at(kf_->id())) {
          traj->append(cf_->t(), cf_->pose());
        }
      }
      kf = map->keyframe();

      kf->computeDerivatives();
      kf->computePcl();
      auto featureMask = std::make_shared<FeatureMask>(kf, 2 * featureSelection->gridSize());
      featureSelection->select(
        kf, [featureMask, lfScore](auto uv1, auto f, auto level) { return (*featureMask)(uv1, f, level) || (*lfScore)(uv1, f, level); });
      f->pose() = motionModel->predict(f->t());
      print("Aligning {} to {}\n", kf->id(), f->id());
      AlignmentRgbd::Results::ConstShPtr results = directIcp->align(kf, f);
      f->pose() = results->pose;
      score = std::make_shared<ScoreAlignment>(f, results, 2 * featureSelection->gridSize());
      featureTracking->track(kf, f, *score);
      entropyRef = std::log(f->pose().cov().determinant());
    }
    motionModel->update(f->pose(), f->t());
    traj->append(dl->timestamps()[fId], f->pose().inverse());
    if (lf != kf) {
      map->addFrame(lf);
    }
    lf = f;
    lfScore = score;

    if (fId > tRmse && fId % tRmse == 0) {
      if (thread.joinable()) {
        thread.join();
      }
      try {
        thread = std::thread([&]() {
          evaluation::tum::writeTrajectory(*traj, trajectoryAlgoPath);
          evaluation::computeKPIs(dl->sequenceId(), experimentId, false);
        });
      } catch (const std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
      }
    };
  }
  if (thread.joinable()) {
    thread.join();
  }
  evaluation::tum::writeTrajectory(*traj, trajectoryAlgoPath);
  evaluation::computeKPIs(dl->sequenceId(), experimentId, false);
}
