import os
import glob

from heimdall.rest import resources
from heimdall.models.ClassifierStats import ClassifierStats
from heimdall.tasks import create_classifier, load_classifier


def init_models(app, db):
    print("init_models")

    # Resync database on startup
    resources.resync_db()

    # Check for dlib models and download if necessary
    resources.check_models()

    db_model = ClassifierStats.query.filter_by(loaded=True).first()

    if db_model is None:
        app.clf = create_classifier()
        path = app.config['ML_MODEL_PATH'] + os.sep + '*.pkl'
        modelList = glob.glob(path)
        if modelList:
            latest_model = max(modelList, key=os.path.getctime)
            app.clf = load_classifier(latest_model)
            db_model = ClassifierStats.query.order_by(ClassifierStats.date.desc()).first()
            if db_model:  # todo pre-compiled pkl files won't be recognized here on first startup...
                app.labels = db_model.labels_as_dict()
                db_model.loaded = True
            db.session.commit()
    else:
        app.clf = load_classifier(db_model.model_path)
        app.labels = db_model.labels_as_dict()