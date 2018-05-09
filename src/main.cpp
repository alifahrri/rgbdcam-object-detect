#include "rgbdcam.h"
#include "yolowrapper.h"
#include "detectionserver.h"
#include <thread>
#include <chrono>
#include <mutex>
#include <ctime>
#include <utility>
#include "utility.hpp"

std::vector<float> readDistance(cv::Mat &m, const Darknet::BBoxes &boxes);

int main(int argc, char **argv)
{
	RGBDCam *rgbdcam;
	if(argc > 1)
	{
		if(std::string(argv[1]) == std::string("record"))
		{
			auto t = std::time(nullptr);
			auto date = std::string(::ctime(&t));
			for(auto &d : date)
				d = ((d == ' ' || d == '\0' || d == '\n') ? '-' : ((d==':') ? '.' : d));
			auto filename = std::string(::getenv("PWD")) + '/' + date + "record.oni";
			if(argc > 2)
				filename = argv[2];
			std::cout << "recording to : " << filename << std::endl;
			rgbdcam = new RGBDCam(filename, true);
		}
		else if(access(argv[1],F_OK) != -1)
			rgbdcam = new RGBDCam(argv[1]);
		else
		{
			std::cerr << ("file does not exist! exiting\n");
			return -1;
		}
	}
	else
		rgbdcam = new RGBDCam();
	rgbdcam->init();
	bool running = true;
	std::mutex mutex;
	// cv::namedWindow("color", CV_WINDOW_NORMAL);
	// cv::namedWindow("depth", CV_WINDOW_NORMAL);
	Darknet darknet;
	DetectionServer detection_server;
	cv::Mat color, depth;

	std::thread darknet_thread([&]
	{
		while(running)
		{
			utility::timer timer;
			mutex.lock();
			auto img = color.clone();
			// auto img = color;
			mutex.unlock();

			if(!img.empty())
			{
				darknet.detect(&img);
				std::cout << "[darknet thread] detecting time : " << timer.elapsed() << std::endl;
			}
			// else
				// std::this_thread::sleep_for(std::chrono::milliseconds(33));
			timer.sleep(33.3);
		}
	});

	while(running)
	{
		utility::timer timer;
		mutex.lock();
		rgbdcam->readMat(color, depth);
		mutex.unlock();
		// std::system("clear");
		auto detections = color.clone();

		auto result = darknet.getResult();
		auto res_it = result.begin();
		auto res_end = result.end();

		std::vector<cv::Point2i> detection_center;
		for(const auto &b : result)
		{
			auto pt = cv::Point2i(b.x + b.w/2, b.y + b.w/2);
			detection_center.push_back(cv::Point2i(pt));
		}
		auto wpts = rgbdcam->convertCoordinate(detection_center);
		auto wpts_it = wpts.begin();
		auto wpts_end = wpts.end();

		auto res_depth = readDistance(depth, result);

		detection_server.publish(result, wpts, res_depth);
		std::stringstream ss;
		// for(const auto &d : res_depth)
		// {
		// 	if(res_it != res_end)
		// 	{
		// 		ss << res_it->label << "(raw : " << std::to_string(d) << ") (prob : " << std::to_string(res_it->confidence) << ") ";
		// 		res_it ++;
		// 	}
		// 	if(wpts_it != wpts_end)
		// 	{
		// 		ss << " pos(" << std::to_string(wpts_it->x) << ", " 
		// 			<< std::to_string(wpts_it->y) << ", " 
		// 			<< std::to_string(wpts_it->z) << ");";
		// 		wpts_it++;
		// 	}
		// }
		// std::cout << "[result : depth] " << ss.str() << std::endl;
		darknet.drawDetections(detections, result);
		cv::imshow("detections", detections);
		cv::imshow("depth", depth);
		// darknet.detect(&color);

		// ss << "\n";
		ss << std::to_string(result.size());
		std::string msg("detected : ");
		msg += ss.str();
		std::cout << msg << std::endl;
		// detection_server.publish(msg);
		auto c = cv::waitKey(1);
		std::cout << "[rgbd loop] elapsed : " << timer.elapsed() << std::endl;
		if(c == 27)
			running = false;
		timer.sleep(33.3);
	}
	darknet_thread.join();
	rgbdcam->cleanUp();
	// cv::destroyAllWindows();
	return 0;
}

std::vector<float> readDistance(cv::Mat &m, const Darknet::BBoxes &boxes)
{
	std::vector<float> ret;
	for(const auto &b : boxes)
	{
		auto pt = cv::Point2i(b.x + b.w/2, b.y + b.w/2);
		auto d = m.at<uchar>(pt);
		ret.push_back(d);
	}
	return ret;
}