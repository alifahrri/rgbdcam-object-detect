#include "rgbdcam.h"
#include "yolowrapper.h"
#include <thread>
#include <chrono>
#include <mutex>

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
			// auto img = color.clone();
			auto img = color;
			mutex.unlock();

			if(!img.empty())
				darknet.detect(&img);
			else
				std::this_thread::sleep_for(std::chrono::milliseconds(33));
		}
	});

	// IplImage ipl;
	while(running)
	{
		mutex.lock();
		rgbdcam.readMat(color, depth);
		mutex.unlock();
		// ipl = color;
		// if(darknet.ipl)
		// 	cvShowImage("ipl", darknet.ipl);
		// if(darknet.ipl_display)
		// 	cvShowImage("result", darknet.ipl_display);
		auto detections = color.clone();
		darknet.drawDetections(&detections);
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
