#include "detectionserver.h"
#include <utility>
#include <iostream>

#define PORT_DETECTION 52746

DetectionServer::DetectionServer()
{
	std::cout << "DetectionServer()\n";
	socket = new QUdpSocket();
	std::cout << "Ready\n";
	// socket->bind(QHostAddress::LocalHost, PORT_DETECTION);
}

void DetectionServer::publish(const Darknet::BBoxes &boxes, const RGBDCam::Point3Ds &world_pts, const std::vector<float> &raw_depth)
{
	object_detection::Detections detections;
	auto wpts_it	 = world_pts.begin();
	auto wpts_end	 = world_pts.end();
	auto raw_it		 = raw_depth.begin();
	auto raw_end	 = raw_depth.end();
	for(const auto &b : boxes)
	{
		object_detection::Detections::Object *obj = detections.add_objects();
		obj->set_class_(b.label);
		obj->set_x_pixel((float)b.x);
		obj->set_y_pixel((float)b.y);
		obj->set_confidence((float)b.confidence);
		if(wpts_it != wpts_end)
		{
			// obj->mutable_x_mm() = (float)wpts_it->x;
			// obj->mutable_y_mm() = (float)wpts_it->y;
			// obj->mutable_z_mm() = (float)wpts_it->z;
			obj->set_x_mm((float)wpts_it->x);
			obj->set_y_mm((float)wpts_it->y);
			obj->set_z_mm((float)wpts_it->z);
			wpts_it++;
		}
		if(raw_it != raw_end)
		{
			obj->set_z_pixel((float)*raw_it);
			raw_it++;
		}
	}
	QByteArray data;
	detections.PrintDebugString();
	data.resize(detections.ByteSize());
	detections.SerializeToArray(data.data(), data.size());
	socket->writeDatagram(data.data(), data.size(), QHostAddress::LocalHost, PORT_DETECTION);
}

void DetectionServer::publish(const std::string &msg)
{
	QByteArray data;
	data.append(msg.c_str());
	socket->writeDatagram(data, QHostAddress::LocalHost, PORT_DETECTION);
}