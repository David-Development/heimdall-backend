SETUP=0
#SETUP=1


cp app/default_config_docker.py app/default_config.py

set DISPLAY :0

if [ "$SETUP" = "1" ]
    # Cleanup
    echo "" > celeryd.log
    rm celeryd.pid

    #echo "init database..."
    #rm -R migrations/
    #python manage.py db init

    #echo "migrate database..."
    #python manage.py db migrate

    #echo "upgrade database..."
    #python manage.py db upgrade
fi


echo "start celery..."
celery worker --detach -A celery_worker.celery --loglevel=DEBUG --logfile celeryd.log

echo "run..."
python manage.py run