#include "detectionserver.h"
#include <iostream>

#define PORT_DETECTION 52746

DetectionServer::DetectionServer()
{
	std::cout << "DetectionServer()\n";
	socket = new QUdpSocket();
	// socket->bind(QHostAddress::LocalHost, PORT_DETECTION);
}

void DetectionServer::publish(const std::string &msg)
{
	QByteArray data;
	data.append(msg.c_str());
	socket->writeDatagram(data, QHostAddress::LocalHost, PORT_DETECTION);
}