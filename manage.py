#!/usr/bin/env python

#import threading
#import os

from app.recognition import RecognitionManager

from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
from app import create_app, db, init_models

app = create_app()
manager = Manager(app)
migrate = Migrate(app, db)

RecognitionManager.init(app, db)


@manager.command
def run():
    init_models()
    app.run(debug=True,
            host='0.0.0.0',
            port=5000,
            use_reloader=True)


manager.add_command('db', MigrateCommand)


if __name__ == '__main__':
    print("Starting app...")
    #for thread in threading.enumerate():
    #    print("PID: ", os.getpid(), " Thread: ", thread.name)
    manager.run()
