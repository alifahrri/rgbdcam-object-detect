#include "yolowrapper.h"
// extern "C"
// {
	#include "darknet.h"
	#include "col2im.c"
	#include "im2col.c"
	#include "deconvolutional_layer.c"
	#include "local_layer.c"
	#include "avgpool_layer.c"
	#include "reorg_layer.c"
	#include "route_layer.c"
	#include "crop_layer.c"
	#include "activation_layer.c"
	#include "shortcut_layer.c"
	#include "normalization_layer.c"
	#include "dropout_layer.c"
	#include "matrix.c"
	#include "cost_layer.c"
	#include "data.c"
	#include "layer.c"
	#include "tree.c"
	#include "maxpool_layer.c"
	#include "detection_layer.c"
	#include "rnn_layer.c"
	#include "region_layer.c"
	#include "network.c"
	#include "batchnorm_layer.c"
	#include "list.c"
	#include "gemm.c"
	#include "softmax_layer.c"
	#include "connected_layer.c"
	#include "lstm_layer.c"
	#include "gru_layer.c"
	#include "convolutional_layer.c"
	#include "crnn_layer.c"
	#include "activations.c"
	#include "option_list.c"
	#include "box.c"
	#include "parser.c"
	#include "utils.c"
	#include "blas.c"
	#include "image.c"
	#include "darknet.h"
// }

#define DEFAULT_AVG 1
#define DEFAULT_CFG "/home/fahri/dev/rgbdcam-object-detect/darknet/cfg/tiny-yolo.cfg"
#define DEFAULT_DATA "/home/fahri/dev/rgbdcam-object-detect/darknet/cfg/coco.data"
#define DEFAULT_WEIGHT "/home/fahri/dev/rgbdcam-object-detect/darknet/tiny-yolo.weights"
#define CAMERA_INDEX 					1
#define DEFAULT_FRAME_SKIP		 0
#define DEFAULT_PREFIX 					0
#define DEFAULT_FPS 						30
#define DEFAULT_WIDTH 					0
#define DEFAULT_HEIGHT 					0
#define DEFAULT_THRESH 				 0.5
#define DEFAULT_HIER 					  0.5
#define DEFAULT_CLASS 					20

Darknet::Darknet()
{
	std::cout << "Darknet() :" << std::endl;
	char* cfg_file = DEFAULT_CFG;
	char* weight_file = DEFAULT_WEIGHT;
	auto alphabet = load_alphabet();
	auto options = read_data_cfg(DEFAULT_DATA);
	int classes = option_find_int(options, "classes", DEFAULT_CLASS);
	auto names = get_labels(option_find_str(options, "names", "/home/fahri/dev/rgbcam-object-detect/darknet/data/names.list"));
	
	this->demo_frame = DEFAULT_AVG;
	this->predictions = (float**)calloc(this->demo_frame,sizeof(float*));
	this->demo_names = names;
	this->demo_alphabet = alphabet;
	this->demo_thresh = DEFAULT_THRESH;
	this->demo_hier = DEFAULT_HIER;
	this->demo_index = CAMERA_INDEX;
	this->demo_done = 0;
	this->demo_classes = classes;
	std::cout << "load network (" << cfg_file << ") (" << weight_file << ")\n";
	this->net = load_network(cfg_file,weight_file,0);
	set_batch_network(net,1);
	this->buff_index = 0;
	this->demo_index = 0;

	auto l = this->net->layers[net->n-1];
	this->demo_detections = l.n*l.w*l.h;
	this->avg = (float *) calloc(l.outputs, sizeof(float));
	for(int j = 0; j < demo_frame; ++j) 
		this->predictions[j] = (float *) calloc(l.outputs, sizeof(float));
	this->boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
	this->boxes_result = (box *) calloc(l.w*l.h*l.n, sizeof(box));

    this->probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
	this->probs_result = (float **) calloc(l.w*l.h*l.n, sizeof(float*));
    for(int j = 0; j < l.w*l.h*l.n; ++j) 
	{
		this->probs[j] = (float *)calloc(l.classes+1, sizeof(float));
		this->probs_result[j] = (float *)calloc(l.classes+1, sizeof(float));
	}
}

void Darknet::ipl_into_image(IplImage *src, image im)
{
	unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
}

image Darknet::ipl_to_image(IplImage *src)
{
	int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image out = make_image(w, h, c);
    ipl_into_image(src, out);
    return out;
}

image Darknet::mat_to_image(cv::Mat *mat)
{
	if(!ipl)
		ipl = cvCreateImage(cvSize(mat->cols,mat->rows), IPL_DEPTH_8U, mat->channels());
	auto m = mat->clone();
	*ipl = m;
	return ipl_to_image(ipl);
}

void Darknet::detect(cv::Mat *mat)
{
	std::cout << "detecting cv::Mat\n";
	float nms = .4;
	auto l = this->net->layers[this->net->n-1];

	std::cout << "convert image.. " << std::endl;
	if(!ipl)
		ipl = cvCreateImage(cvSize(mat->cols,mat->rows), IPL_DEPTH_8U, mat->channels());
	if(m.empty())
		m.create(mat->cols, mat->rows, CV_8UC3);
	mat->copyTo(m);
	*ipl = m;
	auto im = ipl_to_image(ipl);
	buff[buff_index] = copy_image(im);
	// free_image(im);
	std::cout << "done" << std::endl;

	std::cout << "letterbox image.. " << std::endl;
	// buff_letter[buff_index] = letterbox_image(im, this->net->w, this->net->h);
	auto boxed = letterbox_image(buff[buff_index],net->w,net->h);
	// auto X = buff_letter[buff_index].data;
	auto X = boxed.data;
	std::cout << "done" << std::endl;

	std::cout << "predict.. " << std::endl;
	auto prediction = network_predict(this->net, X);
	std::cout << "done" << std::endl;

	std::cout << "memcopy.. " << std::endl;
	memcpy(this->predictions[this->demo_index], prediction, l.outputs*sizeof(float));
	mean_arrays(this->predictions, this->demo_frame, l.outputs, this->avg);
	l.output = avg;
	std::cout << "done" << std::endl;

	if(l.type == DETECTION)
	{
		std::cout << "last layer type : Detection\n";
		get_detection_boxes(l, 1, 1, this->demo_thresh, this->probs, boxes, 0);
	}
	else if(l.type == REGION)
	{
		std::cout << "last layer type : Region\n";
		get_region_boxes(l, buff[0].w, buff[0].h, net->w, net->h, this->demo_thresh, this->probs, this->boxes, 0, 0, 0, demo_hier, 1);
	}
	else 
	{
		error("Last layer must produce detections\n");
	}

	do_nms_obj(this->boxes, this->probs, l.w*l.h*l.n, l.classes, nms);

	mutex.lock();
	// *(this->boxes_result) = *(this->boxes);
	for(int i=0; i<l.w*l.h*l.n; i++)
		this->boxes_result[i] = this->boxes[i];
	for(int i=0; i < demo_detections; i++)
		for(int j=0; j<demo_classes; j++)
			this->probs_result[i][j] = this->probs[i][j];
	mutex.unlock();

	// std::cout << "drawing.. ";
	// printf("\033[2J");
    // printf("\033[1;1H");
    // printf("Objects:\n\n");
	// // auto display = buff[buff_index];
	// display = buff[buff_index];
	// draw_detections(display, demo_detections, this->demo_thresh, this->boxes, this->probs, 0, demo_names, this->demo_alphabet, this->demo_classes);
	// std::cout << "done" << std::endl;

	// if(!ipl_display)
	// 	ipl_display = cvCreateImage(cvSize(mat->cols,mat->rows), IPL_DEPTH_8U, mat->channels());

	// auto step = ipl_display->widthStep;
	// for(auto y = 0; y<display.h; ++y)
	// 	for(auto x = 0; x<display.w; ++x)
	// 		for(auto k = 0; k<display.c; ++k)
	// 		{
	// 			ipl_display->imageData[y*step + x*display.c + k] = (unsigned char)(get_pixel(display, x, y, k)*255);
	// 		}

	free_image(boxed);
	free_image(im);
	// free_image(display);
}

void Darknet::drawDetections(cv::Mat *mat)
{
	// printf("\033[2J");
    // printf("\033[1;1H");
    // printf("Objects:\n\n");

	// auto img = cvCreateImage(cvSize(mat->cols,mat->rows), IPL_DEPTH_8U, mat->channels());
	// IplImage img = (*mat);
	IplImage im = (*mat);
	IplImage *img = cvCloneImage(&im);
	auto display = ipl_to_image(img);
	// // auto display = buff[buff_index];
	// auto b = *(this->boxes);
	// auto p = *(this->probs);
	mutex.lock();
	draw_detections(display, demo_detections, this->demo_thresh, this->boxes_result, this->probs_result, 0, demo_names, this->demo_alphabet, this->demo_classes);
	mutex.unlock();

	// auto step = img->widthStep;
	// for(auto y = 0; y<display.h; ++y)
	// 	for(auto x = 0; x<display.w; ++x)
	// 		for(auto k = 0; k<display.c; ++k)
	// 		{
	// 			img->imageData[y*step + x*display.c + k] = (unsigned char)(get_pixel(display, x, y, k)*255);
	// 		}
	
	// auto step = ipl_display->widthStep;
	// /*
	auto step = mat->step;
	for(auto y = 0; y<display.h; ++y)
		for(auto x = 0; x<display.w; ++x)
			for(auto k = 0; k<display.c; ++k)
			{
				mat->data[y*step + x*display.c + k] = (unsigned char)(get_pixel(display, x, y, k)*255);
			}
	// */
	cvReleaseImage(&img);
	free_image(display);
}