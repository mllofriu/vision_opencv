/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014, Kei Okada.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Kei Okada nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

// http://github.com/Itseez/opencv/blob/master/samples/cpp/tutorial_code/TrackingMotion/goodFeaturesToTrack_Demo.cpp
/**
 * @function goodFeaturesToTrack_Demo.cpp
 * @brief Demo code for detecting corners using Shi-Tomasi method
 * @author OpenCV team
 */

#include <ros/ros.h>
#include "opencv_apps/nodelet.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <dynamic_reconfigure/server.h>
#include "opencv_apps/FeatureDetectorConfig.h"
#include "opencv_apps/Frame.h"
#include "opencv_apps/KeyPoint.h"
#include "opencv_apps/Point2DArrayStamped.h"

namespace feature_detector {
class FeatureDetectorNodelet : public opencv_apps::Nodelet
{
	image_transport::Publisher img_pub_;
	image_transport::Subscriber img_sub_;
	image_transport::CameraSubscriber cam_sub_;
	ros::Publisher msg_pub_;

	boost::shared_ptr<image_transport::ImageTransport> it_;

	feature_detector::FeatureDetectorConfig config_;
	dynamic_reconfigure::Server<feature_detector::FeatureDetectorConfig> srv;

	bool debug_view_;
	bool publish_image_;
	ros::Time prev_stamp_;

	std::string window_name_;
	static bool need_config_update_;

	int num_features_;
	cv::Ptr<cv::FeatureDetector> ORB_detector_;

	void reconfigureCallback(
			feature_detector::FeatureDetectorConfig& new_config,
			uint32_t level) {
		config_ = new_config;
		num_features_ = config_.num_features;
		ORB_detector_ = cv::ORB::create(num_features_);
	}

	const std::string& frameWithDefault(const std::string& frame,
			const std::string& image_frame) {
		if (frame.empty())
			return image_frame;

		return frame;
	}
	void imageCallbackWithInfo(const sensor_msgs::ImageConstPtr& msg,
			const sensor_msgs::CameraInfoConstPtr& cam_info) {
		do_work(msg, cam_info->header.frame_id);
	}

	void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
		do_work(msg, msg->header.frame_id);
	}

	static void trackbarCallback(int, void*) {
		need_config_update_ = true;
//		ORB_detector_ = cv::ORB::create(num_features_);
	}

	void do_work(const sensor_msgs::ImageConstPtr& msg,
			const std::string input_frame_from_msg) {
		try
		{
			// Convert the image into something opencv can handle.
			cv::Mat frame = cv_bridge::toCvShare(msg, msg->encoding)->image;

			// Do the work
			cv::Mat src_gray;
			int numFeaturesTracker = 100;

			if ( frame.channels() > 1 ) {
				cv::cvtColor( frame, src_gray, cv::COLOR_YUV2GRAY_YUYV );
			} else {
				src_gray = frame;
				cv::cvtColor( src_gray, frame, cv::COLOR_GRAY2BGR );
			}

			if( debug_view_) {
				/// Create Trackbars for Thresholds
				cv::namedWindow( window_name_, cv::WINDOW_AUTOSIZE );
				cv::createTrackbar( "Detected Features", window_name_, &num_features_, 1000, trackbarCallback);
				if (need_config_update_) {
					config_.num_features = num_features_;
					srv.updateConfig(config_);
					need_config_update_ = false;
				}
			}

			/// void goodFeaturesToTrack_Demo( int, void* )
			if( num_features_ < 1 ) { num_features_ = 1; }

			/// Parameters for Shi-Tomasi algorithm
			std::vector<cv::KeyPoint> keypoints;
			cv::Mat descriptors;

			/// Apply corner detection
			cv::Mat mask;
			NODELET_INFO_STREAM("Images dimension: "<< src_gray.size());
			ORB_detector_->detectAndCompute(src_gray,cv::noArray(), keypoints, descriptors);

			/// Draw corners detected
			NODELET_INFO_STREAM("** Number of features detected: "<< keypoints.size());

			//-- Show what you got
			if( publish_image_){
				int r = 4;
				for( size_t i = 0; i < keypoints.size(); i++ )
				{
					cv::circle( frame, keypoints[i].pt, r, cv::Scalar(255, 0, 0));
				}
				// Publish the image.
				sensor_msgs::Image::Ptr out_img = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
				img_pub_.publish(out_img);
			}

			if( debug_view_) {

				cv::imshow( window_name_, frame );
				int c = cv::waitKey(1);
			}

			// Create msgs
			opencv_apps::Frame f;
			for( size_t i = 0; i< keypoints.size(); i++ ) {
				opencv_apps::KeyPoint kp;
				kp.x = keypoints[i].pt.x;
				kp.y = keypoints[i].pt.y;
				kp.size = keypoints[i].size;
				kp.angle = keypoints[i].angle;
				kp.octave = keypoints[i].octave;

				// Pointer to the i-th row
				std::copy(descriptors.ptr<uchar>(i), descriptors.ptr<uchar>(i) + 32, kp.descriptor.begin());

				f.keypoints.push_back(kp);
			}
			f.header.stamp = msg->header.stamp;

			// Publish keypoints
			msg_pub_.publish(f);
		}
		catch (cv::Exception &e)
		{
			NODELET_ERROR("Image processing error: %s %s %s %i", e.err.c_str(), e.func.c_str(), e.file.c_str(), e.line);
		}

		prev_stamp_ = msg->header.stamp;
	}

	void subscribe()
	{
		NODELET_DEBUG("Subscribing to image topic.");
		if (config_.use_camera_info)
			cam_sub_ = it_->subscribeCamera("image", 3, &FeatureDetectorNodelet::imageCallbackWithInfo, this);
		else
			img_sub_ = it_->subscribe("image", 3, &FeatureDetectorNodelet::imageCallback, this);
	}

	void unsubscribe()
	{
		NODELET_DEBUG("Unsubscribing from image topic.");
		img_sub_.shutdown();
		cam_sub_.shutdown();
	}

public:
	virtual void onInit()
	{
		Nodelet::onInit();
		it_ = boost::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(*nh_));

		pnh_->param("debug_view", debug_view_, false);
		pnh_->param("publish_image_",publish_image_, false);

		if (debug_view_) {
			always_subscribe_ = true;
		}
		prev_stamp_ = ros::Time(0, 0);

		window_name_ = "Image";
		num_features_ = 23;

		float scale_factor_ = 1.2f;
		int nlevels_ = 8;
		int edge_threshold_ = 31;
		//	ORB_detector_ = new cv::ORB(num_features_,scale_factor_,nlevels_,edge_threshold_);
		ORB_detector_ = cv::ORB::create(num_features_);
		dynamic_reconfigure::Server<feature_detector::FeatureDetectorConfig>::CallbackType f =
				boost::bind(&FeatureDetectorNodelet::reconfigureCallback, this, _1, _2);
		srv.setCallback(f);

		if ( publish_image_ ){
			img_pub_ = advertiseImage(*pnh_, "image", 1);
		}

		msg_pub_ = advertise<opencv_apps::Frame>(*pnh_, "features", 1);

		onInitPostProcess();
	}
};
bool FeatureDetectorNodelet::need_config_update_ = false;
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(feature_detector::FeatureDetectorNodelet, nodelet::Nodelet);
