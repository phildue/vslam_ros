#include "core/types.h"
#include "log.h"
INITIALIZE_EASYLOGGINGPP
#include <filesystem>

namespace fs = std::filesystem;
namespace vslam::log
{
static std::map<std::string, Config::ShPtr> configs = {{"default", std::make_shared<Config>()}};
void initialize(const std::string & folder, bool clean)
{
  if (clean) {
    fs::remove_all(folder);
  }
  fs::create_directories(folder);
  el::Loggers::reconfigureAllLoggers(
    el::ConfigurationType::Filename, format("{}/vslam.log", folder));
  el::Loggers::reconfigureLogger("performance", el::ConfigurationType::ToStandardOutput, "false");
  el::Loggers::reconfigureLogger("performance", el::ConfigurationType::ToFile, "true");
  el::Loggers::reconfigureLogger(
    "performance", el::ConfigurationType::Filename, format("{}/runtime.log", folder));
}

void create(const std::string & name)
{
  el::Loggers::getLogger(name);
  el::Configurations defaultConf;
  defaultConf.setToDefault();
  defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
  defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
  el::Loggers::reconfigureLogger(name, defaultConf);
  configs.insert({name, std::make_shared<Config>(*configs["default"])});
}

Config::ShPtr config(const std::string & name)
{
  if (configs.find(name) == configs.end()) {
    log::create(name);
  }
  return configs.at(name);
}

void show(const std::string & name, const cv::Mat & mat, int delay)
{
  cv::imshow(name, mat);
  cv::waitKey(delay);
}

}  // namespace vslam::log