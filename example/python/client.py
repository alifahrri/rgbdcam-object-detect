from PyQt5 import QtNetwork as qnet
from PyQt5.QtNetwork import QUdpSocket as udp
from PyQt5.QtCore import QCoreApplication as core
from PyQt5.QtCore import QTimer
import signal
import sys

port = 52746
address = qnet.QHostAddress.LocalHost

app = None
socket = udp()

def read_datagram() :
    size = socket.pendingDatagramSize()
    data = socket.readDatagram(size)
    print data

def sigint_handler(*args) :
    app.quit()
    print 'sigint catched'

if __name__ == '__main__' :
    print 'hello'
    signal.signal(signal.SIGINT, sigint_handler)
    app = core(sys.argv)
    if len(sys.argv) > 1 :
        port = int(sys.argv[1])
    socket = udp()
    socket.bind(address, port)
    socket.readyRead.connect(read_datagram)
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda : None)
    sys.exit(app.exec_())