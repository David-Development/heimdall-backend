import logging
import numpy as np
import os
from PIL import Image
import dlib
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def load_dataset(path, grayscale=True, pil=False):
    c = 0
    X, y, folder_names = [], [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            # Ignore empty folders
            if len(os.listdir(subject_path)) > 0:
                folder_names.append(subdirname)
                for filename in os.listdir(subject_path):
                    if not filename.endswith('.md') and os.path.isfile(os.path.join(subject_path, filename)):
                        if pil:
                            im = Image.open(os.path.join(subject_path, filename))
                            if grayscale:
                                im = im.convert("L")
                        else:
                            im = cv2.imread(os.path.join(subject_path, filename))
                            if grayscale:
                                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                            else:
                                # use a sane color order
                                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        im = np.asarray(im, dtype=np.uint8)
                        im.setflags(write=True)
                        X.append(im)
                        # y.append(subdirname + "_" + os.path.splitext(filename)[0])
                        y.append(c)
                c += 1
    return [X, y, folder_names]


def load_image(path, grayscale=False):
    im = cv2.imread(path)
    if grayscale:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def dlib2cv(rect):
    if isinstance(rect, dlib.rectangle):
        x = rect.left()
        y = rect.top()
        w = rect.width()
        h = rect.height()
        return [x, y, w, h]
    else:
        raise TypeError("rect must be of type dlib.rectangle")


def cv2dlib(rect):
    return dlib.rectangle(long(rect[0]), long(rect[1]), long(rect[0] + rect[2]), long(rect[1] + rect[3]))


def enlarge_rectangle(img, rect, percent):
    return_rect = [rect[0] - int(rect[0] * percent), rect[1] - int(rect[1] * percent),
                   rect[2] + int(rect[2] * percent * 2),
                   rect[3] + int(rect[3] * percent * 2)]
    if return_rect[0] < 0:
        return_rect[0] = 0
    if return_rect[1] < 0:
        return_rect[0] = 0
    if return_rect[0] + return_rect[2] >= img.shape[0]:
        return_rect[2] = img.shape[0] - return_rect[0]
    if return_rect[1] + return_rect[3] >= img.shape[1]:
        return_rect[3] = img.shape[1] - return_rect[1]
    return return_rect


def croprect(image, rect):
    x, y, w, h = [v for v in rect]
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    return image[y: y + h, x:x + w]


def resize_keep_aspect(image, width):
    r = float(width) / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def resize_array(images, size):
    return [cv2.resize(image, (size, size)) for image in images]


def resize_array_keep_aspect(images, width):
    return [resize_keep_aspect(image, width) for image in images]


def rgb_array_to_grayscale(images):
    return [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images]


def write_images_dict_to_folder(images, folder_name):
    c = 0
    for label, image in images.iteritems():
        # cv2.imwrite(os.path.join(folder_name + str(label)) + ".jpg", image)
        Image.fromarray(image).save(os.path.join(folder_name + str(label)) + "_" + str(c) + ".jpg")
        c += 1


def write_images_to_folder(images, labels, folder_name):
    for idx, img in enumerate(images):
        # img = img[:, :, 0]
        # cv2.imwrite(os.path.join(folder_name + str(labels[idx])) + ".jpg", img)
        Image.fromarray(img).save(os.path.join(folder_name + str(labels[idx]) + "_" + str(idx)) + ".jpg")


def to_vector_list(X):
    return [np.ravel(x) for x in X]


def create_label_dict(y, labels):
    return dict(zip(list(xrange(max(y) + 1)), labels))
