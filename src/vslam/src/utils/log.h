#ifndef VSLAM_LOG_H__
#define VSLAM_LOG_H__
#include <Eigen/Dense>
#include <opencv2/highgui.hpp>

#include "easylogging++.h"
namespace vslam::log {
struct Config {
  typedef std::shared_ptr<Config> ShPtr;
  int show = -1;
  int save = -1;
};

void initialize(const std::string &folderOut, bool clean = false);
void configure(const std::string &configFile);
void create(const std::string &name);

void show(const std::string &name, const cv::Mat &mat, int delay = 0);

Config::ShPtr config(const std::string &name);

template <typename Drawable> void show(const std::string &name, const Drawable &drawable, int delay = 0) {
  cv::imshow(name, drawable());
  cv::waitKey(delay);
}

#ifndef LOG_DISABLE_IMAGE_LOGS
template <typename Drawable> void append(const std::string &name, const Drawable &drawable) {
  auto _config = config(name);
  if (_config->show >= 0 || _config->save >= 0) {
    const cv::Mat mat = drawable();
    if (_config->save >= 0) {
      LOG(WARNING) << "Image save not implemented!";
    }
    if (_config->show >= 0) {
      cv::imshow(name, mat);
      cv::waitKey(_config->show);
    }
  }
}
#else
template <typename Drawable> void append(const std::string &name, const Drawable &drawable) {}
#endif

}  // namespace vslam::log
#endif