from heimdall.camera.camera import Camera
from PIL import Image as PILImage

im = PILImage.new("RGB", (800, 600), "black")
im.save('/live_view.jpg')
Camera.load_image_path('/live_view.jpg')
