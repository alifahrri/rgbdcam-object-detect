#ifndef RGBDCAM_H

#define RGBDCAM_H
#include <openni2/OpenNI.h>
#include <opencv2/opencv.hpp>

class RGBDCam
{
public :
	struct Point3D
	{
		double x;
		double y;
		double z;
	};
	typedef std::vector<Point3D> Point3Ds;
public :
	RGBDCam();
	~RGBDCam();
	RGBDCam(const std::string &file);
	void init();
	void readMat(cv::Mat &color, cv::Mat &depth);
	Point3D convertCoordinate(const cv::Point2i &pt);
	Point3Ds convertCoordinate(const std::vector<cv::Point2i> &pts);
private :
	void open_file(const std::string &path);
	void open_device();
	void init_openni();
	void init_stream();
	void start_stream();
private :
	std::string oni_file;
	cv::Mat color_mat;
	cv::Mat depth_mat;
	openni::Device device;
	openni::VideoStream depth_stream;
	openni::VideoStream color_stream;
	openni::VideoFrameRef color_frame;
	openni::VideoFrameRef depth_frame;
};

#endif