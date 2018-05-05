#include "yolowrapper.h"
#include <iostream>
#include <cstring>
#include <unistd.h>

#define DEFAULT_AVG 		(1)
#define DEFAULT_CFG 		"/home/fahri/dev/rgbdcam-object-detect/darknet/cfg/tiny-yolo.cfg"
#define DEFAULT_DATA 		"/home/fahri/dev/rgbdcam-object-detect/darknet/cfg/coco.data"
#define DEFAULT_WEIGHT 		"/home/fahri/dev/rgbdcam-object-detect/darknet/weights/tiny-yolo.weights"
#define CAMERA_INDEX		(1)
#define DEFAULT_FRAME_SKIP	(0)
#define DEFAULT_PREFIX		(0)
#define DEFAULT_FPS			(30)
#define DEFAULT_WIDTH		(0)
#define DEFAULT_HEIGHT		(0)
#define DEFAULT_THRESH		(0.5)
#define DEFAULT_HIER		(0.5)
#define DEFAULT_CLASS		(20)
#define DEFAULT_NAME_LIST	"/home/fahri/dev/rgbcam-object-detect/darknet/data/names.list"

extern "C" 
{
	int size_network(network *net);
	void remember_network(network *net);
	void get_detection_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);
	void get_region_boxes(layer l, int w, int h, int netw, int neth, float thresh, float **probs, box *boxes, float **masks, int only_objectness, int *map, float tree_thresh, int relative);
	void yolo2_do_nms_obj(box *boxes, float **probs, int total, int classes, float thresh);
	detection *avg_predictions(network *net, int *nboxes);
	// void load_network_into(network **net, char *cfg, char *weights, int clear);
	// network* load_network(char *cfg, char *weights, int clear);
	// void set_batch_network(network *net, int b);
	// void load_weights_upto(network *net, char *filename, int start, int cutoff);
	// network *make_network(int n);
}

Darknet::Darknet()
{
	std::cout << "Darknet() :" << std::endl;
	char* cfg_file = DEFAULT_CFG;
	char* weight_file = DEFAULT_WEIGHT;

	std::cout << "loading alphabet : " << std::flush;
	auto alphabet = load_alphabet();
	std::cout << "done!" << std::endl;

	std::cout << "reading data config : " << std::flush;
	auto options = read_data_cfg(DEFAULT_DATA);
	std::cout << "loading classes :\n" << std::flush;
	int classes = option_find_int(options, "classes", DEFAULT_CLASS);
	std::cout << "done!" << std::endl;
	std::string label_str(option_find_str(options, "names", DEFAULT_NAME_LIST));
	char filename[250] = {0};
#ifdef DARKNET_DIR
	if(access(label_str.c_str(), F_OK))
		label_str = std::string(DARKNET_DIR) + "data/coco.names";
#endif
	std::strncpy(filename, label_str.data(), sizeof(char)*label_str.size());
	std::cout << "loading labels : " << filename << std::endl;
	auto names = get_labels(filename);
	std::cout << "done!" << std::endl;
	std::cout << "reading data config : done!" << std::endl;

	// net = (network*)calloc(1, sizeof(network));
	// this->net = new network;
	
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
	// load_net(cfg_file, weight_file);
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

	demo_total = size_network(net);
	this->avg = (float *) calloc(demo_total, sizeof(float));
	this->predictions = (float**)calloc(demo_frame, sizeof(float*));
	for (int i = 0; i < demo_frame; ++i){
        predictions[i] = (float*)calloc(demo_total, sizeof(float));
    }
}

// void Darknet::load_weight(char *weight)
// {
// 	std::cout << "loading weights.. " << std::endl;
// 	load_weights_upto(net, weight, 0, net->n);
// 	std::cout << "loading weights.. done" << std::endl;
// }

// void Darknet::load_net(char *cfg, char* weight)
// {
// 	std::cout << "loading network\n";
// 	parse_net(cfg);
// 	std::cout << "network size : " << this->net->n << std::endl;
// 	for(int i=0; i<net->n; i++)
//         printf("layer[%d].type = %d\n", i, net->layers[i].type);
// 	load_weight(weight);
// 	std::cout << "net ptr : " << net << std::endl;
// 	for(size_t i=0; i<net->n; i++)
// 	{
// 		auto l = net->layers[i];
// 		std::cout << "layeridx : " << i << " type : " << l.type << std::endl;
// 	}
// 	std::cout << "loading network done\n";
// }


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

// detection* Darknet::avgPredictions(network *net, int *nboxes, float **predictions)
// {
// 	int i, j;
//     int count = 0;
//     fill_cpu(demo_total, 0, avg, 1);
//     for(j = 0; j < demo_frame; ++j){
//         axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
//     }
//     for(i = 0; i < net->n; ++i){
//         layer l = net->layers[i];
//         if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
//             memcpy(l.output, avg + count, sizeof(float) * l.outputs);
//             count += l.outputs;
//         }
//     }
//     detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
//     return dets;
// }

// void Darknet::rememberNetwork(network *net, float **predictions, int idx, float *forwarded_output)
// {
// 	int i;
//     int count = 0;
//     for(i = 0; i < net->n; ++i){
//         layer l = net->layers[i];
//         if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
//             memcpy(predictions[idx] + count, net->layers[i].output, sizeof(float) * l.outputs);
//             count += l.outputs;
//         }
//     }
// }

// detection *Darknet::boxProbToDetection(box *boxes, float **probs, size_t n)
// {
	
// }

void Darknet::detect(cv::Mat *mat)
{
	std::cout << "detecting... " << std::flush;
	float nms = .4;
	auto last_layer_idx = this->net->n-1;
	auto l = this->net->layers[this->net->n-1];

	if(!ipl)
		ipl = cvCreateImage(cvSize(mat->cols,mat->rows), IPL_DEPTH_8U, mat->channels());

	if(m.empty())
		m.create(mat->cols, mat->rows, CV_8UC3);

	mat->copyTo(m);
	*ipl = m;
	auto im = ipl_to_image(ipl);
	buff[buff_index] = im;
	
	auto boxed = letterbox_image(im, net->w, net->h);
	auto X = boxed.data;
	
	auto prediction = network_predict(this->net, X);
	// network_predict(this->net, X);
	// remember_network(net);
	// rememberNetwork(this->net, this->predictions, demo_index);
	// int nboxes = 0;
	// this->detects_result = avgPredictions(net, &nboxes, this->predictions);
	
	memcpy(this->predictions[this->demo_index], prediction, l.outputs*sizeof(float));
	mean_arrays(this->predictions, this->demo_frame, l.outputs, this->avg);
	l.output = avg;

	if(l.type == DETECTION)
		get_detection_boxes(l, 1, 1, this->demo_thresh, this->probs, boxes, 0);
	else if((l.type == REGION) || (l.type == YOLO))
		get_region_boxes(l, im.w, im.h, net->w, net->h, this->demo_thresh, this->probs, this->boxes, 0, 0, 0, demo_hier, 1);
	else 
	{
		std::cout << "layer type : " << l.type << std::endl;
		error("Last layer must produce detections\n");
	}

	yolo2_do_nms_obj(this->boxes, this->probs, l.w*l.h*l.n, l.classes, nms);
	// do_nms_obj(detects_result, nboxes, l.classes, nms);

	mutex.lock();
#if 1
	for(int i=0; i<l.w*l.h*l.n; i++)
		this->boxes_result[i] = this->boxes[i];
	for(int i=0; i < demo_detections; i++)
		for(int j=0; j<demo_classes; j++)
			this->probs_result[i][j] = this->probs[i][j];
#else
	// for(int i=0; i<l.w*l.h*l.n; i++)
	// 	this->detects_result[i] = dets[i];
#endif
	mutex.unlock();

	free_image(boxed);
	free_image(im);
	std::cout << "detecting... done" << std::endl;
}

std::vector<Darknet::BBox> Darknet::getResult()
{
	std::vector<BBox> boxes;
	for(int i=0; i<demo_detections; i++)
	{
		BBox bbox;
		char labelstr[4096] = {0};
		int class_ = -1;
		for(int j=0; j < demo_classes; j++)
		{
#if 1
			if(probs_result[i][j] > demo_thresh)
#else
			if(this->detects_result[i].prob[j] > demo_thresh)
#endif
			{
				if(class_ < 0)
				{
					strcat(labelstr, demo_names[j]);
					class_ = j;
				}
				else 
				{
					strcat(labelstr,",");
					strcat(labelstr, demo_names[j]);
				}
				bbox.label = std::string(labelstr);
#if 1
				bbox.confidence = probs_result[i][j];
#else
				bbox.confidence = this->detects_result[i].prob[j];
#endif
			}
		}

		if(class_>= 0) 
		{
			auto w = buff[buff_index].w;
			auto h = buff[buff_index].h;
			int width = buff[buff_index].h * 0.006;
			int offset = class_*123457%demo_classes;
			float red = get_color(2, offset, demo_classes);
			float green = get_color(1, offset, demo_classes);
			float blue = get_color(0, offset, demo_classes);
			bbox.rgb[0] = red;
			bbox.rgb[1] = green;
			bbox.rgb[2] = blue;
#if 1
			box b = this->boxes_result[i];
#else
			box b = this->detects_result[i].bbox;
#endif

			bbox.x = (b.x-b.w/2)*w;
			bbox.y = (b.y-b.h/2)*h;
			bbox.w = b.w*w;
			bbox.h = b.h*h;

			boxes.push_back(bbox);
		}
	}
	return boxes;
}

void Darknet::drawDetections(cv::Mat &mat, const BBoxes &bboxes)
{
	// auto bboxes = this->getResult();
	for(const auto& b : bboxes)
	{
		auto color = cv::Scalar(b.rgb[2]*255, b.rgb[1]*255, b.rgb[0]*255);
		cv::rectangle(mat,cv::Rect(b.x,b.y,b.w,b.h),color,2);
		auto text = b.label + " : " + std::to_string(b.confidence);
		cv::putText(mat,text,cv::Point(b.x,b.y),CV_FONT_HERSHEY_PLAIN, 2.0, color, 2);
	}
}