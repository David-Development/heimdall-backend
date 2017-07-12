**Run Postgres Docker**  
`sudo docker run --name postgres -e POSTGRES_PASSWORD=password -d -p 5432:5432 postgres`  
postgres available under `0.0.0.0:5432`

**Access Postgres Docker**  
`sudo docker exec -i -t <name> /bin/bash`

**Run Redis Docker**  
`sudo docker run -d --name redis -p 6379:6379 redis`  
redis available under `0.0.0.0:6379`

**Run Celery Worker**  
`celery worker -A app.celery --loglevel=info`
`