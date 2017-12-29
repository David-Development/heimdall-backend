#!/bin/bash

python="python3" # "python3" or "python"

export PYTHONUNBUFFERED=TRUE
set DISPLAY :0


$python flask_mqtt/setup.py install
$python -m pip install line_profiler


models_file="./heimdall/ml_models/dlib_face_recognition_resnet_model_v1.dat"
if [ -f "$models_file" ]
then
    echo "$models_file exists! Skipping initialization.."
else
    echo "$models_file not found.. Downloading models.. Please wait.."
    cd ./heimdall/ml_models
    wget https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2
    wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
    bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2 # unzip
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2     # unzip
    rm dlib_face_recognition_resnet_model_v1.dat.bz2 # cleanup
    rm shape_predictor_68_face_landmarks.dat.bz2     # cleanup
    cd ../../
    echo "##################################################"
    
fi

setup_file="/setup_done"
if [ -f "$setup_file" ]
then
    echo "$setup_file exists! Skipping initialization.."
else
    echo "##################################################"
    echo "$setup_file file not found. Starting init process!"

    echo "init database..."
    rm -R migrations/
    $python manage.py db init

    echo "migrate database..."
    $python manage.py db migrate

    echo "upgrade database..."
    $python manage.py db upgrade

    echo "initializing database..."
    $python manage.py initialize_database

    echo "" > $setup_file

    echo "done!"
    echo "##################################################"
fi

echo "run..."
$python manage.py run

#gunicorn --bind 0.0.0.0:5000 wsgi // https://stackoverflow.com/questions/33379287/gunicorn-cant-find-app-when-name-changed-from-application
#$python test.py
