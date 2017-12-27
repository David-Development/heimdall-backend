import time
from heimdall.camera.base_camera import BaseCamera
from io import BytesIO
from PIL import Image as PILImage
from queue import Queue
import cv2

q = Queue()

class Camera(BaseCamera):

    @staticmethod
    def load_image(image):
        ret, jpeg = cv2.imencode('.jpg', image)
        q.put(jpeg.tobytes())

    @staticmethod
    def load_image_path(path):
        im = PILImage.open(path)
        # im.thumbnail((w, h), Image.ANTIALIAS)
        io = BytesIO()
        im.save(io, format='JPEG')
        frame = io.getvalue()
        # frame = utils.load_image('/live_view.jpg', False)

        #print("Camera Type1:", type(q))
        q.put(frame)

    @staticmethod
    def frames():
        while True:
            #print("Camera Type2:", type(q))
            time.sleep(0)
            frame = q.get()
            time.sleep(0)
            #print("Yield image!", type(frame))
            yield frame
