#!/bin/bash


setup_file="/setup_done"

python="python3" # "python3" or "python"

cp app/default_config_docker.py app/default_config.py
export PYTHONUNBUFFERED=TRUE
set DISPLAY :0

# Cleanup
echo "" > celeryd.log
rm celeryd.pid

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


#echo "start celery..."
#celery worker --detach -A celery_worker.celery --loglevel=DEBUG --logfile celeryd.log

echo "run..."
$python manage.py run

#$python test.py
