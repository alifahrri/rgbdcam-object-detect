#ifndef CLIENT_H
#define CLIENT_H

#include <QUdpSocket>

class DetectionClient : public QObject
{
    Q_OBJECT
public :
    explicit DetectionClient(quint16 port, QObject *parent = nullptr);
private :
    QUdpSocket *socket;
    QHostAddress sender;
    quint16 port;
public slots : 
    void readyRead();
};

#endif