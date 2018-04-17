#ifndef RGBDCAM_H

#define RGBDCAM_H
#include <openni2/OpenNI.h>
#include <opencv2/opencv.hpp>

class RGBDCam
{
public :
	RGBDCam();
	~RGBDCam();
	void readMat(cv::Mat &color, cv::Mat &depth);
private :
	void open_device();
	void init_stream();
	void start_stream();
private :
	openni::Device device;
	openni::VideoStream depth_stream;
	openni::VideoStream color_stream;
	openni::VideoFrameRef color_frame;
	openni::VideoFrameRef depth_frame;
};

#endif