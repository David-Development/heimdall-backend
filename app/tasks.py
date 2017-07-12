from app import db, app
from models import Gallery, Image
import os
import shutil

config = app.config


def sync_db_from_filesystem():
    clear_files_from_db()
    # start with new and unknown images
    create_and_populate_gallery(config['IMAGE_BASE_PATH'], 'new', False)
    create_and_populate_gallery(config['IMAGE_BASE_PATH'], 'unknown', False)

    # and the subjects
    for gallery_folder in os.listdir(config['SUBJECTS_BASE_PATH']):
        create_and_populate_gallery(config['SUBJECTS_BASE_PATH'], gallery_folder)


def create_and_populate_gallery(base_path, gallery_name, subject_gallery=True):
    gallery_path = os.path.join(base_path, gallery_name)

    if subject_gallery:
        gallery_folder = os.path.join(config['SUBJECT_IMAGES_FOLDER'], gallery_name)
    else:
        gallery_folder = os.path.join(config['IMAGE_FOLDER'], gallery_name)

    gallery = Gallery(name=gallery_name, path=gallery_folder, subject_gallery=subject_gallery)
    db.session.add(gallery)
    db.session.commit()
    for file_path in os.listdir(gallery_path):
        db.session.add(Image(name=file_path, gallery_id=gallery.id, path=os.path.join(gallery_folder, file_path)))
    db.session.commit()


def clear_files_from_db():
    clear_table(Image)
    clear_table(Gallery)
    db.session.commit()


def clear_table(model):
    db.session.query(model).delete()


def delete_gallery(src_gallery, dest_gallery):
    """
    Move Images from source to destination gallery and removes src gallery
    :param src_gallery: the source gallery :type Gallery
    :param dest_gallery: the destination gallery :type Gallery
    :return: 
    """

    # move images
    move_gallery_content(src_gallery=src_gallery, dest_gallery=dest_gallery)

    # delete src gallery
    os.rmdir(os.path.join(config['BASEDIR'], src_gallery.path))


def move_gallery_content(src_gallery, dest_gallery):
    basedir = config['BASEDIR']
    src_path = os.path.join(basedir, src_gallery.path)
    dest_path = os.path.join(basedir, dest_gallery.path)

    for file_path in os.listdir(src_path):
        shutil.move(os.path.join(src_path, file_path), os.path.join(dest_path, file_path))


def move_images(gallery, images):
    """
    Move on or more images into a new gallery
    :param gallery: the new gallery for the images
    :param images: the images to be moved
    :return: 
    """

    basedir = config['BASEDIR']
    dest_gallery_path = os.path.join(basedir, gallery.path)

    for image in images:
        image_path = os.path.join(basedir, image.path)
        shutil.move(image_path, os.path.join(dest_gallery_path, image.name))
