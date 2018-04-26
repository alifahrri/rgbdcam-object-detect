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
	char **demo_names;
	image **demo_alphabet;
	int demo_classes;
	float **probs;
	box *boxes;
	network *net;
	image buff[3];
	image buff_letter[3];
	int buff_index = 0;
	float fps = 0;
	float demo_thresh = 0;
	float demo_hier = .5;
	int running  = 0;
	int demo_frame = 3;	
	int demo_detections = 0;
	float **predictions;
	int demo_index = 0;
	int demo_done = 0;
	float *avg;
	double demo_time;
};

#endif