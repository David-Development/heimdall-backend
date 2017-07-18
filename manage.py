from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
from app import create_app, db, socketio

app = create_app()
migrate = Migrate(app, db)
manager = Manager(app)

manager.add_command('db', MigrateCommand)


@manager.command
def run():
    socketio.run(app,
                 debug=True,
                 host='0.0.0.0',
                 port=5000)


if __name__ == '__main__':
    manager.run()
