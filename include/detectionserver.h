#ifndef DETECTIONSERVER_H
#define DETECTIONSERVER_H

#include <QUdpSocket>
#include <string>
#include "yolowrapper.h"
#include "rgbdcam.h"
#include "protocol/message.pb.h"

class DetectionServer
{
public : 
	DetectionServer();
	void publish(const Darknet::BBoxes &boxes, const RGBDCam::Point3Ds &world_pts, const std::vector<float> &raw_depth);
	void publish(const std::string &msg);
private : 
	QUdpSocket *socket;
};

#endif