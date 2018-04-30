#ifndef RGBDCAM_H

#define RGBDCAM_H
#include <openni2/OpenNI.h>
#include <opencv2/opencv.hpp>

class RGBDCam
{
	enum mode_t {
		STREAM, 
		STREAM_RECORD,
		PLAY
	};

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
	RGBDCam(const std::string &file, bool record = false);

	void init();
	void cleanUp();
	void readMat(cv::Mat &color, cv::Mat &depth);
	Point3D convertCoordinate(const cv::Point2i &pt);
	Point3Ds convertCoordinate(const std::vector<cv::Point2i> &pts);

private :
	void open_file(const std::string &path);
	void open_device();
	void init_openni();
	void init_stream();
	void init_recorder(const std::string &path);
	void start_stream();

private :
	mode_t mode;
	bool record_mode = false;
	std::string oni_file;
	cv::Mat color_mat;
	cv::Mat depth_mat;
	openni::Device device;
	openni::Recorder recorder;
	openni::VideoStream depth_stream;
	openni::VideoStream color_stream;
	openni::VideoFrameRef color_frame;
	openni::VideoFrameRef depth_frame;
};

#endif