#include "rgbdcam.h"
#include "yolowrapper.h"
#include <thread>
#include <chrono>
#include <mutex>

std::vector<float> readDistance(cv::Mat &m, const Darknet::BBoxes &boxes);

int main(int argc, char **argv)
{
	RGBDCam *rgbdcam;
	if(argc > 1)
		rgbdcam = new RGBDCam(argv[1]);
	else
		rgbdcam = new RGBDCam();
	rgbdcam->init();
	bool running = true;
	std::mutex mutex;
	// cv::namedWindow("color", CV_WINDOW_NORMAL);
	cv::namedWindow("depth", CV_WINDOW_NORMAL);
	Darknet darknet;
	cv::Mat color, depth;

	std::thread darknet_thread([&]
	{
		while(running)
		{
			mutex.lock();
			auto img = color.clone();
			// auto img = color;
			mutex.unlock();

			if(!img.empty())
				darknet.detect(&img);
			else
				std::this_thread::sleep_for(std::chrono::milliseconds(33));
		}
	});

	while(running)
	{
		mutex.lock();
		rgbdcam->readMat(color, depth);
		mutex.unlock();
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

		std::stringstream ss;
		for(const auto &d : res_depth)
		{
			if(res_it != res_end)
				(*res_it).label += ((" (" )+std::to_string((int)d)+(") "));
			ss << ((res_it != res_end) ? ((*res_it++).label + " : ") : "") <<  d << "; ";
			if(wpts_it != wpts_end)
			{
				ss << " (" << std::to_string(wpts_it->x) << ", " 
					<< std::to_string(wpts_it->y) << ", " 
					<< std::to_string(wpts_it->z) << ";";
				wpts_it++;
			}
		}
		std::cout << "[result : depth] " << ss.str() << std::endl;
		darknet.drawDetections(detections, result);
		cv::imshow("detections", detections);
		cv::imshow("depth", depth);
		// darknet.detect(&color);
		auto c = cv::waitKey(33);
		if(c == 27)
			running = false;
	}
	darknet_thread.join();
	cv::destroyAllWindows();
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