from threading import Thread, Event
import socket
import time

from app import socketio

thread_stop_event = Event()


class CameraThread(Thread):
    def __init__(self, host, port):
        super(CameraThread, self).__init__()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.s.bind((host, port))
        except socket.error as msg:
            print 'Socket binding failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]

        self.s.listen(5)
        print 'Socket binding complete, waiting for image'

    def check_for_image(self):

        # wait for image
        while not thread_stop_event.isSet():
            conn, addr = self.s.accept()
            print 'Connected with ' + addr[0] + ':' + str(addr[1])
            image = ''
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                image += data

            image = image.decode("hex").encode("base64")
            filename = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())) + '.jpg'
            self.push_image(image)

        self.s.close()

    def push_image(self, image):
        socketio.emit('new_image', {'image': image})

    def run(self):
        print 'thread running'
        self.check_for_image()
