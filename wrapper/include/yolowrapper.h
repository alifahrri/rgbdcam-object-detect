#ifndef YOLOWRAPPER_H
#define YOLOWRAPPER_H

#include "darknet.h"
#include "utils.h"
#include "parser.h"
#include "detection_layer.h"
#include "option_list.h"
#include "image.h"
/*
extern "C"
{
#include "utils.h"
#include "demo.h"
#include "col2im.h"
#include "im2col.h"
#include "deconvolutional_layer.h"
#include "local_layer.h"
#include "avgpool_layer.h"
#include "reorg_layer.h"
#include "route_layer.h"
#include "crop_layer.h"
#include "activation_layer.h"
#include "shortcut_layer.h"
#include "normalization_layer.h"
#include "dropout_layer.h"
#include "matrix.h"
#include "cost_layer.h"
#include "data.h"
#include "layer.h"
#include "tree.h"
#include "maxpool_layer.h"
#include "detection_layer.h"
#include "rnn_layer.h"
#include "region_layer.h"
#include "network.h"
#include "batchnorm_layer.h"
#include "list.h"
#include "gemm.h"
#include "softmax_layer.h"
#include "connected_layer.h"
#include "lstm_layer.h"
#include "gru_layer.h"
#include "convolutional_layer.h"
#include "crnn_layer.h"
#include "activations.h"
#include "option_list.h"
#include "box.h"
#include "parser.h"
#include "utils.h"
#include "blas.h"
#include "image.h"
#include "network.h"
#include "yolo2_box.h"
}
*/

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
	struct BBox 
	{
		int x;
		int y;
		int w;
		int h;
		std::string label;
		float confidence;
		float rgb[3];
	};
	typedef std::vector<BBox> BBoxes;
	
public :
	Darknet();
	void detect(IplImage *ipl);
	void detect(cv::Mat *mat);
	void drawDetections(cv::Mat &mat, const BBoxes &bboxes);
	void run();
	BBoxes getResult();

private :
	void parse_net(char *cfg);
	void load_weight(char *weight);
	void load_net(char* cfg, char* weight);
	void ipl_into_image(IplImage *src, image im);
	image ipl_to_image(IplImage *src);
	image mat_to_image(cv::Mat *mat);
	void rememberNetwork(network *net, float** predictions, int idx, float *forwarded_output = nullptr);
	detection *avgPredictions(network *net, int *nboxes, float **predictions);
	detection *boxProbToDetection(box *boxes, float** probs, size_t n);

public :
	cv::Mat m;
	IplImage *ipl = 0;
	IplImage *ipl_display = 0;

private :
	std::mutex mutex;
	image **demo_alphabet;
	image buff_letter[3];
	image buff[3];
	network *net = nullptr;
	box *boxes;
	box *boxes_result;
	detection *detects;
	detection *detects_result = nullptr;
	float **predictions;
	double demo_time;
	float demo_thresh = 0;
	float demo_hier = .5;
	float fps = 0;
	float **probs;
	float **probs_result;
	float *avg;
	char **demo_names;
	int demo_total = 0;
	int demo_detections = 0;
	int demo_frame = 3;
	int demo_index = 0;
	int demo_done = 0;
	int buff_index = 0;
	int running  = 0;
	int demo_classes;
};

#endif