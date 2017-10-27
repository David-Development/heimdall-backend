cp app/default_config_docker.py app/default_config.py

python manage.py db init
python manage.py db migrate
python manage.py db upgrade
celery worker --detach -A celery_worker.celery --loglevel=DEBUG --logfile celeryd.log
python manage.py run


# Cleanup
echo "" > celeryd.log
rm celeryd.pid