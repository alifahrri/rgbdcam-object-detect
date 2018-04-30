#include "rgbdcam.h"
#include <iostream>

RGBDCam::RGBDCam()
{
	mode = STREAM;
}

RGBDCam::RGBDCam(const std::string &file, bool record)
{
	record_mode = record;
	if(!record_mode)
		mode = PLAY;
	else
		mode = STREAM_RECORD;

	std::cout << (record ? "recording to : " : "opening file : ") << file << std::endl;
	oni_file = file;
}

RGBDCam::~RGBDCam()
{
	
}

void RGBDCam::cleanUp()
{
	std::cout << "clean up.. ";
	depth_stream.destroy();
	color_stream.destroy();
	device.close();
	openni::OpenNI::shutdown();
	std::cout << "done\n";
}

void RGBDCam::init()
{
	this->init_openni();
	switch(mode)
	{
		case STREAM :
			this->open_device();
			break;
		case STREAM_RECORD :
			this->open_device();
			this->init_recorder(oni_file);
			break;
		case PLAY :
			this->open_file(oni_file);
			break;
		default :
			break;
	}
	this->init_stream();
	this->start_stream();
}

void RGBDCam::init_recorder(const std::string &file)
{
	recorder.create(file.c_str());
}

void RGBDCam::init_openni()
{
	if(openni::OpenNI::initialize() != openni::STATUS_OK)
	{
		std::cerr << "OpenNI failed to initialized : "
						<< openni::OpenNI::getExtendedError() 
						<< std::endl;
		openni::OpenNI::shutdown();
		exit(-1);
	}
}

RGBDCam::Point3D RGBDCam::convertCoordinate(const cv::Point2i &pt)
{
	float x = 0.0;
	float y = 0.0;
	float z = 0.0;
	if(abs(pt.x) < depth_mat.cols && abs(pt.y) < depth_mat.rows)
	{
		auto depth_z =  depth_mat.at<openni::DepthPixel>(pt);
		if(depth_z > 0)
			openni::CoordinateConverter::convertDepthToWorld(depth_stream,pt.x, pt.y, depth_z, &x, &y, &z);
	}
	return {x, y, z};
}

RGBDCam::Point3Ds RGBDCam::convertCoordinate(const std::vector<cv::Point2i> &pts)
{
	Point3Ds w_pts;
	for(const auto &p : pts)
		w_pts.push_back(convertCoordinate(p));
	return w_pts;
}

void RGBDCam::readMat(cv::Mat &color, cv::Mat &depth)
{
	color_stream.readFrame(&color_frame);
	depth_stream.readFrame(&depth_frame);

	static const auto size = cv::Size(640,480);

	color_mat = cv::Mat(size,CV_8UC3,(void*)color_frame.getData());
	depth_mat = cv::Mat(size, CV_16UC1, (void*)depth_frame.getData());

	color = color_mat;
	depth = depth_mat;
	depth.convertTo(depth,CV_8U,255./depth_stream.getMaxPixelValue());
	cv::cvtColor(color, color, CV_RGB2BGR);
}

void RGBDCam::start_stream()
{
	color_stream.start();
	depth_stream.start();
	if(record_mode)
	{
		recorder.attach(color_stream);
		recorder.attach(depth_stream);
		recorder.start();
	}
}

void RGBDCam::open_file(const std::string& path)
{
	auto status = device.open(path.c_str());
	if(status != openni::STATUS_OK)
	{
		std::cerr << "Device open failed : "
						<< path << " "
						<< openni::OpenNI::getExtendedError() 
						<< std::endl;
		openni::OpenNI::shutdown();
		exit(-1);
	}
}

void RGBDCam::open_device()
{
	auto status = device.open(openni::ANY_DEVICE);
	if(status != openni::STATUS_OK)
	{
		std::cerr << "Device open failed : "
						<< openni::OpenNI::getExtendedError() 
						<< std::endl;
		openni::OpenNI::shutdown();
		exit(-1);
	}
}

void RGBDCam::init_stream()
{
	auto status = depth_stream.create(device, openni::SENSOR_DEPTH);
	if(status == openni::STATUS_OK)
	{
		openni::VideoMode mode;
		mode.setResolution(640,480);
		mode.setFps(30);
		mode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);

		auto s = depth_stream.setVideoMode(mode);
		if(s != openni::STATUS_OK)
		{
			std::cerr << "Couldn't apply video mode on depth stream : "
							<< openni::OpenNI::getExtendedError()
							<< std::endl;
		}
	}
	else
	{
		std::cerr << "Couldn't find depth stream : "
						<< openni::OpenNI::getExtendedError();
	}

	status = color_stream.create(device, openni::SENSOR_COLOR);
	if(status == openni::STATUS_OK)
	{
		openni::VideoMode mode;
		mode.setResolution(640,480);
		mode.setFps(30);
		mode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);
		status = color_stream.setVideoMode(mode);
		if(status != openni::STATUS_OK)
		{
			std::cerr << "Couldn't apply video mode on depth stream : "
							<< openni::OpenNI::getExtendedError()
							<< std::endl;
		}
	}
	else
	{
		std::cerr << "Couldn't find color stream : "
						<< openni::OpenNI::getExtendedError();
	}

	if(!depth_stream.isValid() || !color_stream.isValid())
	{
		std::cerr << "No valid streams. Exiting." << std::endl;
		exit(-2);
	}
}