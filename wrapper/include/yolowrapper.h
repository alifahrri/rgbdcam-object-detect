#ifndef YOLOWRAPPER_H
#define YOLOWRAPPER_H

// extern "C"
// {
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
// }

#include <mutex>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"
#if CV_MAJOR_VERSION == 3
#include "opencv2/videoio/videoio_c.h"
#include "opencv2/imgcodecs/imgcodecs_c.h"
#endif

class Darknet
{
public :
	Darknet();
	void detect(IplImage *ipl);
	void detect(cv::Mat *mat);
	void drawDetections(cv::Mat *mat);
	void run();

private :
	void ipl_into_image(IplImage *src, image im);
	image ipl_to_image(IplImage *src);
	image mat_to_image(cv::Mat *mat);

public :
	cv::Mat m;
	IplImage *ipl = 0;
	IplImage *ipl_display = 0;

private :
	std::mutex mutex;
	image **demo_alphabet;
	image buff_letter[3];
	image buff[3];
	// image display;
	network *net;
	box *boxes;
	box *boxes_result;
	double demo_time;
	float demo_thresh = 0;
	float demo_hier = .5;
	float fps = 0;
	float **predictions;
	float **probs;
	float **probs_result;
	float *avg;
	char **demo_names;
	int demo_detections = 0;
	int demo_frame = 3;
	int demo_index = 0;
	int demo_done = 0;
	int buff_index = 0;
	int running  = 0;
	int demo_classes;
};

#endif