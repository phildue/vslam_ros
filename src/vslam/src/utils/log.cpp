#include "core/types.h"
#include "log.h"
INITIALIZE_EASYLOGGINGPP
#include <filesystem>

namespace fs = std::filesystem;
namespace vslam::log {
static std::map<std::string, Config::ShPtr> configs = {{"default", std::make_shared<Config>()}};
void initialize(const std::string &folder, bool clean) {
  if (clean) {
    fs::remove_all(folder);
  }
  fs::create_directories(folder);
  el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Filename, format("{}/vslam.log", folder));
  el::Loggers::reconfigureLogger("performance", el::ConfigurationType::ToStandardOutput, "false");
  el::Loggers::reconfigureLogger("performance", el::ConfigurationType::ToFile, "true");
  el::Loggers::reconfigureLogger("performance", el::ConfigurationType::Filename, format("{}/runtime.log", folder));
}
void configure(const std::string &directory) {
  for (const auto &name : {"direct_odometry", "features", "direct_pose_graph"}) {
    const std::string filepath = format("{}/{}.conf", directory, name);
    LOG(INFO) << format("Loading config for {} at {}", name, filepath);
    el::Loggers::reconfigureLogger(name, el::Configurations(filepath));
  }
}

void create(const std::string &name) {
  el::Loggers::getLogger(name);
  configs.insert({name, std::make_shared<Config>(*configs["default"])});
}

Config::ShPtr config(const std::string &name) {
  if (configs.find(name) == configs.end()) {
    log::create(name);
  }
  return configs.at(name);
}

void show(const std::string &name, const cv::Mat &mat, int delay) {
  cv::imshow(name, mat);
  cv::waitKey(delay);
}

}  // namespace vslam::log