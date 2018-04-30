#ifndef DETECTIONSERVER_H
#define DETECTIONSERVER_H

#include <QUdpSocket>
#include <string>

class DetectionServer
{
public : 
	DetectionServer();
	void publish(const std::string &msg);
private : 
	QUdpSocket *socket;
};

#endif