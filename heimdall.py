#!flask/bin/python
from app import app, socketio

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')

