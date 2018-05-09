#include "client.h"
#include <iostream>
#include <sstream>
#include <utility>

DetectionClient::DetectionClient(quint16 port, QObject *parent)
    : QObject(parent), 
      socket(new QUdpSocket(this))
{
    this->port = port;
    sender = QHostAddress::LocalHost;
    socket->bind(sender,port);
    connect(this->socket, SIGNAL(readyRead()),
            this, SLOT(readyRead()));
    std::cout << "Ready\n";
}

void DetectionClient::readyRead()
{
    object_detection::Detections detections;
    QByteArray data;
    data.resize(socket->pendingDatagramSize());
    socket->readDatagram(data.data(), data.size(), &sender, &port);
    detections.ParseFromArray(data.data(), data.size());
    std::string str = detections.DebugString();
    detected_objects.clear();
    for(int i=0; i<detections.objects_size(); i++)
    {
        const object_detection::Detections::Object &obj = detections.objects(i);
        Detection object;
        object.label = obj.class_();
        object.x_pixel = obj.x_pixel();
        object.y_pixel = obj.y_pixel();
        object.z_pixel = obj.z_pixel();
        object.x_pixel = obj.x_pixel();
        object.y_pixel = obj.y_pixel();
        object.z_pixel = obj.z_pixel();
        object.confidence = obj.confidence();
        detected_objects.push_back(object);
        // std::cout << "object " << i << ": " << obj.class_()
        //           << " (" 
        //           << obj.x_pixel() << ","
        //           << obj.y_pixel() << ","
        //           << obj.z_pixel() 
        //           << ") ("
        //           << obj.x_mm() << ","
        //           << obj.y_mm() << ","
        //           << obj.z_mm() 
        //           << ") confidence : " 
        //           << obj.confidence()
        //           << "\n";
    }
    detections.PrintDebugString();
}