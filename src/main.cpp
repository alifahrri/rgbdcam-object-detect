#include "rgbdcam.h"

int main(int argc, char **argv)
{
	RGBDCam rgbdcam;
	bool running = true;
	cv::namedWindow("color", CV_WINDOW_NORMAL);
	cv::namedWindow("depth", CV_WINDOW_NORMAL);
	while(running)
	{
		cv::Mat color, depth;
		rgbdcam.readMat(color, depth);
		cv::imshow("color", color);
		cv::imshow("depth", depth);
		auto c = cv::waitKey(33);
		if(c == 27)
			running = false;
	}
	cv::destroyAllWindows();
	return 0;
}
