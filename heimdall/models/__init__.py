import os
import glob


def init_models(app, db):
    from heimdall.rest import resources
    from heimdall.models.ClassifierStats import ClassifierStats
    from heimdall.tasks import create_classifier, load_classifier

    print("init_models")

    # Resync database on startup
    resources.resync_db()

    # Check for dlib models and download if necessary
    resources.check_models()

    db_model = ClassifierStats.query.filter_by(loaded=True).first()
    app.labels = {}

    if db_model is None:
        app.clf = create_classifier()
        path = app.config['ML_MODEL_PATH'] + os.sep + '*.pkl'
        model_list = glob.glob(path)
        if model_list:
            latest_model = max(model_list, key=os.path.getctime)
            print("Loading model:", latest_model)
            app.clf = load_classifier(latest_model)
            db_model = ClassifierStats.query.order_by(ClassifierStats.date.desc()).first()
            if db_model:  # todo pre-compiled pkl files won't be recognized here on first startup...
                app.labels = db_model.labels_as_dict()
                db_model.loaded = True
            db.session.commit()
    else:
        app.clf = load_classifier(db_model.model_path)
        app.labels = db_model.labels_as_dict()
