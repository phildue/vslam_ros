//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include <utils/utils.h>
#include "Loss.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::least_squares;
TEST(LossTest, TukeyLoss)
{

  std::vector<double> l, l_, r, w;
  auto tk = std::make_shared<TukeyLoss>();
  for (int i = -100; i < 100; i++) {
    const double ri = (double)i / 10.0;
    r.push_back(ri);
    l.push_back(tk->compute(ri));
    l_.push_back(tk->computeDerivative(ri));
    w.push_back(tk->computeWeight(ri) * ri * ri);

  }

#ifdef TEST_VISUALIZE
  vis::plt::figure();
  vis::plt::title("TukeyLoss");
  vis::plt::named_plot("l(r)", r, l);
  vis::plt::named_plot("l'(r)", r, l_);
  vis::plt::named_plot("$w(r)r^2$'", r, w);
  vis::plt::legend();
  vis::plt::xlabel("r");
  vis::plt::show();
#endif
}
