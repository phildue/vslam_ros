//
// Created by phil on 07.08.21.
//

#ifndef VSLAM_LOG_H
#define VSLAM_LOG_H

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "easylogging++.h"
#include "core/core.h"
#include "visuals.h"

//TODO user should define this
#define SYSTEM(loglevel) CLOG(loglevel, "system")
#define IMAGE_ALIGNMENT(loglevel) CLOG(loglevel, "image_alignment")
#define SOLVER(loglevel) CLOG(loglevel, "solver")

#define LOG_IMG(name) pd::vslam::Log::getImageLog(name, pd::vslam::Level::Debug)
#define LOG_PLT(name) pd::vslam::Log::getPlotLog(name, pd::vslam::Level::Debug)


namespace pd::vslam {
  class Frame;
  class FrameRGBD;
  using Level = el::Level;

  class LogPlot
  {
public:
    typedef std::shared_ptr < LogPlot > ShPtr;

    LogPlot(const std::string & file, bool block = false, bool show = true, bool save = false);
    void setHeader(const std::vector < std::string > & header);
    void append(vis::Plot::ConstShPtr plot)
    {
      if (_show || _save) {
        plot->plot();
        if (_show) {
          vis::plt::show(_block);
        }
        if (_save) {
          vis::plt::save(_name + ".jpg");
        }
      }
    }
    bool _block;
    bool _show;
    bool _save;

private:
    const std::string _name;

  };

  void operator << (LogPlot::ShPtr log, vis::Plot::ConstShPtr plot);

  template < typename ... Args >
  using DrawFunctor = cv::Mat (*)(Args... args);

  class LogImage
  {
public:
    typedef std::shared_ptr < LogImage > ShPtr;

    LogImage(const std::string & name, bool block = false, bool show = true, bool save = false);
    template < typename ... Args >
    void append(DrawFunctor < Args... > draw, Args... args)
    {
      if (_show || _save) {
        logMat(draw(args ...));
      }
    }
    void append(const cv::Mat & mat)
    {
      if (_show || _save) {
        logMat(mat);
      }
    }
    void append(vis::Drawable::ConstShPtr drawable)
    {
      if (_show || _save) {
        logMat(drawable->draw());
      }
    }
    template < typename T >
    void append(const Eigen::Matrix < T, -1, -1 > & mat)
    {
      if (_show || _save) {
        logMat(vis::drawAsImage(mat.template cast < double > ()));
      }
    }
    bool _block;
    bool _show;
    bool _save;

protected:
    const std::string _name;
    const std::string _folder;
    std::uint32_t _ctr;

    void logMat(const cv::Mat & mat);

  };

  template < typename T >
  void operator << (LogImage::ShPtr log, const Eigen::Matrix < T, -1, -1 > &mat) {
    log->append(mat);
    }
    void operator << (LogImage::ShPtr log, vis::Drawable::ConstShPtr drawable);

  class Log {
public:
    static std::shared_ptr < Log > get(
      const std::string & name, const std::string & configFilePath,
      Level level = el::Level::Info);
    static std::shared_ptr < LogImage > getImageLog(
      const std::string & name,
      Level level = el::Level::Info);
    static std::shared_ptr < LogPlot > getPlotLog(const std::string & name, Level level);
    static Level _showLevel;
    static Level _blockLevel;

    Log(const std::string & name, const std::string & configFilePath);

private:
    const std::string _name;
    static std::map < std::string, std::map < Level, std::shared_ptr < Log >> > _logs;
    static std::map < std::string, std::map < Level, std::shared_ptr < LogPlot >> > _logsPlot;
    static std::map < std::string, std::map < Level, std::shared_ptr < LogImage >> > _logsImage;

  };
}

#endif //VSLAM_LOG_H
