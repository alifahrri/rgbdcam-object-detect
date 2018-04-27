#include "rgbdcam.h"
#include "yolowrapper.h"
#include <thread>
#include <chrono>
#include <mutex>

std::vector<float> readDistance(cv::Mat &m, const Darknet::BBoxes &boxes);

int main(int argc, char **argv)
{
	RGBDCam rgbdcam;
	bool running = true;
	std::mutex mutex;
	cv::namedWindow("color", CV_WINDOW_NORMAL);
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
		rgbdcam.readMat(color, depth);
		mutex.unlock();
		auto detections = color.clone();
		auto result = darknet.getResult();
		auto res_depth = readDistance(depth, result);
		std::stringstream ss;
		auto res_it = result.begin();
		auto res_end = result.end();
		for(const auto &d : res_depth)
		{
			ss << ((res_it != res_end) ? ((*res_it++).label + " : ") : "") <<  d << "; ";
		}
		std::cout << "result depth : " << ss.str() << std::endl;
		darknet.drawDetections(detections, result);
		cv::imshow("color", detections);
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