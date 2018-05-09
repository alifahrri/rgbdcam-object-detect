#include <QtCore>
#include "client.h"

#define PORT_DETECTION 52746

int main(int argc, char **argv)
{
    QCoreApplication app(argc,argv);
    DetectionClient detect_client(PORT_DETECTION);
    return app.exec();
}