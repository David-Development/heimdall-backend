from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
from app import create_app, db, socketio, init_models


app = create_app()
manager = Manager(app)
migrate = Migrate(app, db)

@manager.command
def run():
    init_models()
    socketio.run(app,
                 debug=True,
                 host='0.0.0.0',
                 port=5000)


manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    print("Starting app...")
    manager.run()
