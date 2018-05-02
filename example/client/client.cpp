#include "client.h"
#include <iostream>

DetectionClient::DetectionClient(quint16 port, QObject *parent)
    : QObject(parent), 
      socket(new QUdpSocket(this))
{
    this->port = port;
    sender = QHostAddress::LocalHost;
    socket->bind(sender,port);
    connect(this->socket, SIGNAL(readyRead()),
            this, SLOT(readyRead()));
}

void DetectionClient::readyRead()
{
    QByteArray data;
    data.resize(socket->pendingDatagramSize());
    socket->readDatagram(data.data(), data.size(), &sender, &port);

    std::cout << "receive data : " << data.toStdString() << std::endl;
}