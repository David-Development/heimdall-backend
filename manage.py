#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import threading
#import os
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
from heimdall.app import create_app

app = create_app()
manager = Manager(app)

from heimdall.app import db
from heimdall.models import init_models
migrate = Migrate(app, db)

@manager.command
def run():
    init_models(app, db)

    app.run(host='0.0.0.0',
            port=5000,
            use_reloader=False,
            threaded=True,  # otherwise only one client can connect to mjpeg stream
            debug=True)


manager.add_command('db', MigrateCommand)


if __name__ == '__main__':
    print("Starting heimdall...")
    #for thread in threading.enumerate():
    #    print("PID: ", os.getpid(), " Thread: ", thread.name)
    manager.run()
