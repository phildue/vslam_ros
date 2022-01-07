//
#include "rosbag2_cpp/readers/sequential_reader.hpp"
#include "rosbag2_cpp/reader.hpp"
#include <iostream>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <nav_msgs/msg/path.hpp>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "vslam_ros/vslam_ros.h"


using namespace testing;
using namespace pd;
using namespace pd::vision;


class LukasKanadeSE3Test : public TestWithParam<int>{
        public:
        std::unique_ptr<rosbag2_cpp::readers::SequentialReader> _reader;
        Camera::ShPtr _camera;
        std::vector<Image> _imgs;
        std::vector<DepthMap> _depths;
        const double _scale = 1.0;
        nav_msgs::msg::Path _pathImu,_pathVo;
        Sophus::SE3d _pose;
        Eigen::Vector6d _xInit;
        int _fNoStart = 0;
        int _nFrames;
        rcutils_time_point_value_t _tStart = 0;
        bool _visualize = false;
        std::vector<double> _errTrans,_errAng;

        LukasKanadeSE3Test(){
                _reader = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
                rosbag2_storage::StorageOptions storageOptions;
                storageOptions.uri = "/media/data/dataset/rgbd_dataset_freiburg1_xyz/rgbd_dataset_freiburg1_xyz.db3";
                storageOptions.storage_id = "sqlite3";
                rosbag2_cpp::ConverterOptions converterOptions;
                _reader->open(storageOptions,converterOptions);
                _fNoStart = GetParam();

                std::vector<rosbag2_storage::TopicMetadata> meta = _reader->get_all_topics_and_types();
                //std::cout << "Found: " << _reader->get_metadata().topics_with_message_count.size() << " meta entries.";
              
                for (const auto& mc : _reader->get_metadata().topics_with_message_count)
                {
                      //  std::cout << mc.topic_metadata.name << ": " << mc.topic_metadata.type << std::endl;

                        if(mc.topic_metadata.name == "/camera/rgb/image_color")
                        {
                                _nFrames = mc.message_count;
                        }
                }
                if (_fNoStart > _nFrames)
                {
                        throw std::runtime_error("Start frame is higher than total amount of frames: " + std::to_string(_fNoStart) + " > " + std::to_string(_nFrames)+ "!");
                }
                //Estimated from gt:
                Eigen::MatrixXd covar(6,6);
                covar << 0.0004, -0.,      0.,      0.,      0.0003,  0.0002,
                -0.,      0.0003, -0.,     -0.0001, -0.,      0.,
                0.,     -0.,      0.0004, -0.0002, -0.,      0.,
                0.,     -0.0001, -0.0002,  0.0007,  0.,     -0.,
                0.0003, -0.,     -0.,      0.,      0.0006,  0.0001,
                0.0002,  0.,      0.,     -0.,      0.0001,  0.0003;


                _xInit << random::N(covar);
                loadData();
        }
        ~LukasKanadeSE3Test(){
                _reader->close();
        }

        void loadData()
        {
                int fNo = 0;

               while(_reader->has_next() && (_camera == nullptr || _depths.size() < 2 || _imgs.size() < 1))
                {

                        rosbag2_storage::SerializedBagMessageSharedPtr msg =  _reader->read_next();
                        //std::cout << fNo << "/" << _nFrames << " t: " << msg->time_stamp << " topic: " << msg->topic_name << std::endl;
                        if ( fNo < _fNoStart || msg->time_stamp < _tStart )
                        {
                                 if(msg->topic_name == "/camera/rgb/image_color")
                                {
                                        fNo++;
                                }
                                continue;
                        }
                        if(msg->topic_name == "/camera/rgb/image_color")
                        {
                                fNo++;
                                
                                sensor_msgs::msg::Image msgDes;
                                rclcpp::SerializedMessage extracted_serialized_msg(*msg->serialized_data);
                                rclcpp::Serialization<sensor_msgs::msg::Image> serialization;
                                serialization.deserialize_message(&extracted_serialized_msg, &msgDes);
                                auto cvImage = cv_bridge::toCvCopy(msgDes);
                                cv::Mat matGray;
                                //cv::resize(cvImage->image,matGray,cv::Size(640,480));
                                cv::cvtColor(cvImage->image,matGray,cv::COLOR_RGB2GRAY);
                                if (_visualize)
                                {
                                        cv::imshow("RGB",cvImage->image);
                                        cv::waitKey(10);
                                }
                                
                                Image img;
                                cv::cv2eigen(matGray,img);
                                img = algorithm::resize(img,_scale);
                                _imgs.push_back(img);

                        }

                       

                        if (msg->topic_name == "/camera/rgb/camera_info")
                        {

                                sensor_msgs::msg::CameraInfo camInfo;
                                rclcpp::SerializedMessage extracted_serialized_msg(*msg->serialized_data);
                                rclcpp::Serialization<sensor_msgs::msg::CameraInfo> serialization;
                                serialization.deserialize_message(&extracted_serialized_msg, &camInfo);

                                _camera = vslam_ros2::convert(camInfo);
                                _camera->resize(_scale);
                        }
                     
                        if(msg->topic_name == "/camera/depth/image")
                        {

                                sensor_msgs::msg::Image msgDes;
                                rclcpp::SerializedMessage extracted_serialized_msg(*msg->serialized_data);
                                rclcpp::Serialization<sensor_msgs::msg::Image> serialization;
                                serialization.deserialize_message(&extracted_serialized_msg, &msgDes);
                                 auto cvImage = cv_bridge::toCvCopy(msgDes);
                                Eigen::MatrixXd depthd;
                                cv::cv2eigen(cvImage->image,depthd);
                                depthd = depthd.array().isNaN().select(0,depthd);
                                depthd = algorithm::resize(depthd,_scale);
                                
                                if (_visualize){
                                        cv::imshow("Depth",cvImage->image);
                                        cv::waitKey(10);
                                        std::cout << depthd.minCoeff() << " --> " << depthd.maxCoeff() << std::endl;
                                }
                                
                                _depths.push_back(depthd);

                        }
                        
                } 
                std::cout << "Data loaded." << std::endl;

        }

        void computeErrorVector(const Eigen::Matrix<double, Eigen::Dynamic, WarpSE3::nParameters>& x)
        {
                const int n =  _errTrans.size();
                _errTrans.resize(n + x.rows());
                _errAng.resize(n + x.rows());
                for (int i = 0; i < x.rows(); i++)
                {
                       
                        _errTrans[n + i] = std::sqrt( std::pow(x(i,0),2) + std::pow(x(i,1),2) + std::pow(x(i,2),2) );
                        _errAng[n + i] = transforms::rad2deg(std::sqrt( std::pow(x(i,3),2) + std::pow(x(i,4),2) + std::pow(x(i,5),2) ));

                        
                }
        }

        void plot(bool block = true)
        {
                if(_visualize)
                {
                        vis::plt::figure();
                        vis::plt::named_plot("Translational Error [m]",_errTrans);
                        vis::plt::legend();
                        vis::plt::figure();
                        vis::plt::named_plot("Angular Error [°]",_errAng);
                        vis::plt::legend();
                        
                        vis::plt::show(block);   
                }
        }
        
};

TEST_P(LukasKanadeSE3Test,DISABLED_LukasKanadeSE3ForwardAdditiveGN)
{
        auto mat0 = vis::drawMat(_imgs[0]);
        auto mat1 = vis::drawMat(_imgs[0]);
        
        Log::getImageLog("I")->append(mat0);
        Log::getImageLog("T")->append(mat1);
        Log::getImageLog("Image Warped")->_block = false;


        auto w = std::make_shared<WarpSE3>(_xInit,_depths[0],_camera);
        auto gn = std::make_shared<GaussNewton<LukasKanadeSE3>> ( 
                        0.3,
                        1e-3,
                        100);
        auto lk = std::make_shared<LukasKanadeSE3> (_imgs[0],_imgs[0],w);
        
        
        //ASSERT_GT(w->x().norm(), 0.1);

   
        gn->solve(lk);
        Eigen::Matrix<double, Eigen::Dynamic, WarpSE3::nParameters> x = gn->x();
        x.conservativeResize(gn->iteration(),Eigen::NoChange);
        computeErrorVector(x);

        EXPECT_LE(_errTrans[_errTrans.size()-1], 0.1);
        EXPECT_LE(_errAng[_errAng.size()-1], 1);


        plot(false);
        
}

TEST_P(LukasKanadeSE3Test, DISABLED_LukasKanadeSE3InverseCompositionalGN)
{
        auto mat0 = vis::drawMat(_imgs[0]);
        auto mat1 = vis::drawMat(_imgs[0]);
        Log::_blockLevel = Level::Unknown;
        Log::_showLevel = Level::Debug;
        Log::getImageLog("I")->append(mat0);
        Log::getImageLog("T")->append(mat1);
        Log::getImageLog("Image Warped")->_block = false;
        //Log::getPlotLog("SolverGN",Level::Debug)->_block = true;
        Log::getPlotLog("SolverGN",Level::Debug)->_show = true;
        Log::getPlotLog("ErrorDistribution",Level::Debug)->_show = false;
        Log::getPlotLog("ErrorDistribution",Level::Debug)->_block = false;

        auto w = std::make_shared<WarpSE3>(_xInit,_depths[0],_camera);
        auto gn = std::make_shared<GaussNewton<LukasKanadeInverseCompositional<WarpSE3>>> ( 
                        1.0,
                        1e-3,
                        150);
        auto lk = std::make_shared<LukasKanadeInverseCompositional<WarpSE3>> (_imgs[0],_imgs[0],w,std::make_shared<TukeyLoss>());
        
        
        //ASSERT_GT(w->x().norm(), 0.1);

 
        gn->solve(lk);
        Eigen::Matrix<double, Eigen::Dynamic, WarpSE3::nParameters> x = gn->x();
        x.conservativeResize(gn->iteration(),Eigen::NoChange);
        computeErrorVector(x);

        EXPECT_LE(_errTrans[_errTrans.size()-1], 0.1) ;
        EXPECT_LE(_errAng[_errAng.size()-1], 1);

        plot();
}

TEST_P(LukasKanadeSE3Test, LukasKanadeSE3InverseCompositionalGNMultiLevel)
{
        auto mat0 = vis::drawMat(_imgs[0]);
        auto mat1 = vis::drawMat(_imgs[0]);
        Log::_blockLevel = Level::Unknown;
        Log::_showLevel = Level::Info;
        Log::getImageLog("I")->append(mat0);
        Log::getImageLog("T")->append(mat1);
        Log::getImageLog("Image Warped")->_block = false;
        //Log::getPlotLog("SolverGN",Level::Debug)->_block = true;
        Log::getPlotLog("SolverGN",Level::Debug)->_show = false;
        Log::getPlotLog("ErrorDistribution",Level::Debug)->_show = false;
        Log::getPlotLog("ErrorDistribution",Level::Debug)->_block = false;

        std::cout << "x0: " << _xInit.transpose() << std::endl;

        auto w = std::make_shared<WarpSE3>(_xInit,_depths[0],_camera);
        auto l = std::make_shared<HuberLoss>(5);
        auto solver = std::make_shared<GaussNewton<LukasKanadeInverseCompositional<WarpSE3>>> ( 
                        1.0,
                        1e-4,
                        20);
        
        for(int i = 4; i > 0; i--)
        {
                const auto s = 1.0/(double)i;
                
                auto templScaled = algorithm::resize(_imgs[0],s);
                auto imageScaled = templScaled;
                auto wScaled = std::make_shared<WarpSE3>(w->x(),algorithm::resize(w->depth(),s),Camera::resize(_camera,s));

                auto lk = std::make_shared<LukasKanadeInverseCompositional<WarpSE3>> (
                        templScaled,
                        imageScaled,
                        wScaled,l);

                solver->solve(lk);
                
                w->setX(wScaled->x());

             
                Eigen::Matrix<double, Eigen::Dynamic, WarpSE3::nParameters> x = solver->x();
                x.conservativeResize(solver->iteration(),Eigen::NoChange);
                
                computeErrorVector(x);

                plot(false);
               
            
        }
        std::cout << "Translational Error: " << _errTrans[_errTrans.size()-1] << "/" << _errTrans[0] << std::endl;
        std::cout << "Angular Error: " << _errAng[_errAng.size()-1] << "/" << _errAng[0] << std::endl;

        EXPECT_LT(_errTrans[_errTrans.size()-1], _errTrans[0] );
        EXPECT_LT(_errAng[_errAng.size()-1], _errAng[0] );


       
}

TEST_P(LukasKanadeSE3Test,DISABLED_LukasKanadeSE3InverseCompositionalLM)
{
        auto mat0 = vis::drawMat(_imgs[0]);
        auto mat1 = vis::drawMat(_imgs[0]);
        Log::_blockLevel = Level::Unknown;
        Log::_showLevel = Level::Debug;
        Log::getImageLog("I")->append(mat0);
        Log::getImageLog("T")->append(mat1);
        Log::getImageLog("Image Warped")->_block = false;
        //Log::getPlotLog("SolverLM",Level::Debug)->_block = true;
        Log::getPlotLog("SolverLM",Level::Debug)->_show = true;

        auto w = std::make_shared<WarpSE3>(_xInit,_depths[0],_camera);
        auto solver = std::make_shared<LevenbergMarquardt<LukasKanadeInverseCompositional<WarpSE3>>> ( 
                        200,
                        1e-9,
                        1e-9,
                        1e-13,
                        1e13);
        auto lk = std::make_shared<LukasKanadeInverseCompositional<WarpSE3>> (_imgs[0],_imgs[0],w);
        
        
        //ASSERT_GT(w->x().norm(), 0.1);

        Eigen::VectorXd chiSquared(solver->maxIterations());
        chiSquared.setZero();
        Eigen::VectorXd chi2pred(solver->maxIterations());
        chi2pred.setZero();
       
        Eigen::VectorXd stepSize(solver->maxIterations());
        stepSize.setZero();
        Eigen::Matrix<double, Eigen::Dynamic, WarpSE3::nParameters> x(solver->maxIterations(),WarpSE3::nParameters);
        x.setConstant(100);
        Eigen::VectorXd lambda(solver->maxIterations());
        lambda.setZero();
        
        solver->solve(lk,chiSquared,chi2pred,stepSize,lambda,x);


        computeErrorVector(x);

        EXPECT_LE(_errTrans[solver->maxIterations()-1], 0.1);
        EXPECT_LE(_errAng[solver->maxIterations()-1], 1);

        plot();
}

INSTANTIATE_TEST_CASE_P(Instantiation, LukasKanadeSE3Test,testing::Range(1,700,15));
/*
INSTANTIATE_TEST_CASE_P(Instantiation, LukasKanadeSE3Test, std::make_tuple(2,"/media/data/dataset/rgbd_dataset_freiburg1_xyz/rgbd_dataset_freiburg1_xyz.db3",1));
INSTANTIATE_TEST_CASE_P(Instantiation, LukasKanadeSE3Test, std::make_tuple(3,"/media/data/dataset/rgbd_dataset_freiburg1_xyz/rgbd_dataset_freiburg1_xyz.db3",1));
INSTANTIATE_TEST_CASE_P(Instantiation, LukasKanadeSE3Test, std::make_tuple(4,"/media/data/dataset/rgbd_dataset_freiburg1_xyz/rgbd_dataset_freiburg1_xyz.db3",1));
INSTANTIATE_TEST_CASE_P(Instantiation, LukasKanadeSE3Test, std::make_tuple(5,"/media/data/dataset/rgbd_dataset_freiburg1_xyz/rgbd_dataset_freiburg1_xyz.db3",1));
*/