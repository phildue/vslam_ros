#pragma once
#include "core/Frame.h"
#include "core/macros.h"
namespace vslam::loop_closure_detection::candidate_proposal {

class CandidateProposal {
public:
  TYPEDEF_PTR(CandidateProposal)

  virtual bool isCandidate(Frame::ConstShPtr f0, Frame::ConstShPtr f1) const = 0;
};
class RelativePose : public CandidateProposal {
public:
  TYPEDEF_PTR(RelativePose)
  RelativePose(double maxTranslation) :
      _maxTranslation(maxTranslation) {}
  bool isCandidate(Frame::ConstShPtr f0, Frame::ConstShPtr f1) const override;

private:
  const double _maxTranslation;
};

class ReprojectedFeatures : public CandidateProposal {
public:
  TYPEDEF_PTR(ReprojectedFeatures)
  ReprojectedFeatures(int minFeatures) :
      _minFeatures(minFeatures) {}
  bool isCandidate(Frame::ConstShPtr f0, Frame::ConstShPtr f1) const override;

private:
  int _minFeatures;
};

}  // namespace vslam::loop_closure_detection::candidate_proposal
