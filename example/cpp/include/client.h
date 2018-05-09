#ifndef CLIENT_H
#define CLIENT_H

#include <QUdpSocket>
#include "message.pb.h"

struct Detection
{
    std::string label;
    int x_pixel;
    int y_pixel;
    int z_pixel;
    float x_mm;
    float y_mm;
    float z_mm;
    float confidence;
};

typedef std::vector<Detection> Detections;

class DetectionClient : public QObject
{
    Q_OBJECT
public :
    explicit DetectionClient(quint16 port, QObject *parent = nullptr);
    virtual ~DetectionClient() {}
    DetectionClient() {}
    Detections detections() { return detected_objects; }
private :
    QUdpSocket *socket;
    QHostAddress sender;
    quint16 port;
    Detections detected_objects;
public slots : 
    void readyRead();
};

#endif