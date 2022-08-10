#include <execution>
#include <vector>

#include "utils/utils.h"
#include "core/core.h"
#include "InverseCompositionalStacked.h"
namespace pd::vslam::lukas_kanade
{

InverseCompositionalStacked::InverseCompositionalStacked(
  const std::vector<std::shared_ptr<InverseCompositional>> & frames)
: least_squares::Problem(frames[0]->nParameters()),
  _frames(frames)
{}

void InverseCompositionalStacked::updateX(const Eigen::VectorXd & dx)
{
  std::for_each(_frames.begin(), _frames.end(), [&dx](auto f) {f->updateX(dx);});
}
void InverseCompositionalStacked::setX(const Eigen::VectorXd & x)
{
  std::for_each(_frames.begin(), _frames.end(), [&x](auto f) {f->setX(x);});
}
least_squares::NormalEquations::ConstShPtr InverseCompositionalStacked::computeNormalEquations()
{
  std::vector<least_squares::NormalEquations::ConstShPtr> nes(_frames.size());
  std::transform(
    _frames.begin(), _frames.end(), nes.begin(), [&](auto f) {
      return f->computeNormalEquations();
    });
  auto ne = std::make_shared<least_squares::NormalEquations>(_frames[0]->nParameters());
  std::for_each(nes.begin(), nes.end(), [&](auto n) {ne->combine(*n);});
  return ne;
}

}
