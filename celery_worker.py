from celery import Celery
from flask_socketio import SocketIO


def make_celery(app):
    celery = Celery(app.import_name, backend=app.config['CELERY_RESULT_BACKEND'],
                    broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    app.socketio = SocketIO(message_queue='redis://')
    return celery
