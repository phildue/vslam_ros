//
// Created by phil on 07.08.21.
//
#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#include "Log.h"

INITIALIZE_EASYLOGGINGPP


namespace pd::vslam
{
std::map<std::string, std::map<Level, std::shared_ptr<Log>>> Log::_logs = {};
std::map<std::string, std::map<Level, std::shared_ptr<LogPlot>>> Log::_logsPlot = {};
std::map<std::string, std::map<Level, std::shared_ptr<LogImage>>> Log::_logsImage = {};
Level Log::_blockLevel = Level::Unknown;
Level Log::_showLevel = Level::Unknown;

std::shared_ptr<Log> Log::get(
  const std::string & name, const std::string & confFilePath,
  Level level)
{
  auto it = _logs.find(name);
  if (it != _logs.end()) {
    return it->second[level];
  } else {
    std::map<Level, std::shared_ptr<Log>> log = {
      {el::Level::Debug, std::make_shared<Log>(name, confFilePath)},
      {el::Level::Info, std::make_shared<Log>(name, confFilePath)},
      {el::Level::Warning, std::make_shared<Log>(name, confFilePath)},
      {el::Level::Error, std::make_shared<Log>(name, confFilePath)},

    };
    _logs[name] = log;
    return log[level];
  }

}

std::shared_ptr<LogImage> Log::getImageLog(const std::string & name, Level level)
{
  auto it = _logsImage.find(name);
  if (it != _logsImage.end()) {
    return it->second[level];
  } else {
    const std::vector<Level> levels = {
      Level::Debug,
      Level::Info,
      Level::Warning,
      Level::Error
    };
    std::map<Level, std::shared_ptr<LogImage>> log;
    for (const auto & l : levels) {
      log[l] = std::make_shared<LogImage>(name, l >= _blockLevel, l >= _showLevel);
    }

    _logsImage[name] = log;
    return log[level];
  }
}
std::shared_ptr<LogPlot> Log::getPlotLog(const std::string & name, Level level)
{
  auto it = _logsPlot.find(name);
  if (it != _logsPlot.end()) {
    return it->second[level];
  } else {
    const std::vector<Level> levels = {
      Level::Debug,
      Level::Info,
      Level::Warning,
      Level::Error
    };
    std::map<Level, std::shared_ptr<LogPlot>> log;
    for (const auto & l : levels) {
      log[l] = std::make_shared<LogPlot>(name, l >= _blockLevel, l >= _showLevel);
    }
    _logsPlot[name] = log;
    return log[level];
  }
}


Log::Log(const std::string & name, const std::string & configFilePath)
: _name(name)
{
  el::Configurations config(configFilePath);
  // Actually reconfigure all loggers instead
  el::Loggers::reconfigureLogger(name, config);

}

LogPlot::LogPlot(const std::string & name, bool block, bool show, bool save)
: _block(block),
  _show(show),
  _save(save),
  _name(name)
{

}
void operator<<(LogPlot::ShPtr log, vis::Plot::ConstShPtr plot)
{
  log->append(plot);
}

LogImage::LogImage(const std::string & name, bool block, bool show, bool save)
: _block(block),
  _show(show),
  _save(save),
  _name(name),
  _folder(LOG_DIR "/" + name),
  _ctr(0U)
{}


void LogImage::logMat(const cv::Mat & mat)
{
  if (mat.cols == 0 || mat.rows == 0) {
    throw pd::Exception("Image is empty!");
  }
  if (_show) {
    cv::imshow(_name, mat);
    // cv::waitKey(_blockLevel <= _blockLevelDes ? -1 : 30);
    cv::waitKey(_block ? 0 : 30);
  }
  if (_save) {
    cv::imwrite(_folder + "/" + _name + std::to_string(_ctr++) + ".jpg", mat);
  }

}

void operator<<(LogImage::ShPtr log, vis::Drawable::ConstShPtr drawable)
{
  log->append(drawable);
}
}
