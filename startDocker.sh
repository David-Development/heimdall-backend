#!/bin/bash


setup_file="/setup_done"

python="python3" # "python3" or "python"

export PYTHONUNBUFFERED=TRUE
set DISPLAY :0


$python flask_mqtt/setup.py install

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

    echo "" > $setup_file

    echo "done!"
    echo "##################################################"
fi

echo "run..."
$python manage.py run
#gunicorn --bind 0.0.0.0:5000 wsgi // https://stackoverflow.com/questions/33379287/gunicorn-cant-find-app-when-name-changed-from-application
#$python test.py
