# Heimdall 
Face recognition prototype with webinterface. Created for the Masterthesis "Neuronale Netze und andere Verfahren zur 
Gesichtserkennung im Heimautomatisierungsfumeld" by Constantin Kirsch for the Master of Science in Computer Science at 
the Bonn-Rhine-Sieg University. Created at the [Multimedia Communication Laboratory](http://mc-lab.inf.h-brs.de/).  
  
 
## Installation


##### OpenCV 
change make -j8 to max number of parallel processes  
remove CUDA/CUBLAS parameters if not applicable
```bash
sudo apt-get install cmake
sudo apt-get install pkg-config
sudo apt-get install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk-3-dev libatlas-base-dev gfortran
sudo apt-get install python2.7-dev

mkdir ~/opencv
cd ~/opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/3.2.0.zip && unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.2.0.zip && unzip opencv_contrib.zip
cd opencv-3.2.0/
mkdir build
cd build/
cmake -D CMAKE_BUILD_TYPE=RELEASE \
   -D CMAKE_INSTALL_PREFIX=/usr/local \
   -D WITH_CUDA=ON \
   -D ENABLE_FAST_MATH=1 \
   -D CUDA_FAST_MATH=1 \
   -D WITH_CUBLAS=1 \
   -D INSTALL_PYTHON_EXAMPLES=ON \
   -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.2.0/modules \
   -D BUILD_EXAMPLES=ON ..

make -j8
sudo make install
sudo ldconfig
```

##### Change to corresponding virtualenv path
```
cd ~/envs/<virtuelenv>/lib/python2.7/site-packages/
ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so
```
##### Install Requirements 
```python -r requirements.txt```

##### Install and start Postgres and Redis 
...  
...  
...

##### Alternatively just use Docker
##### Run Postgres Docker
`sudo docker run --name postgres -e POSTGRES_PASSWORD=password -d -p 5432:5432 postgres`  
postgres available under `0.0.0.0:5432`

##### Access Postgres Docker
`sudo docker exec -i -t <name> /bin/bash`

##### Run Redis Docker
`sudo docker run -d --name redis -p 6379:6379 redis`  
redis available under `0.0.0.0:6379`

##### Configure Application
- Copy `default_config_template.py` to `default_config.py`
- Adjust values inside `default_config.py`

##### Initialize Database
`python manage.py db init`  
`python manage.py db migrate`  
`python manage.py db upgrade`


##### Run Celery Worker
`celery worker -A celery_worker.celery --loglevel=info`

##### Run Application
`python manage.py run`

## Usage
Create desired folder structure under "app/images/subjects/" folder. Insert images and call `/api/resync` to populate 
the database. Alternatively create the galleries in the Webinterface. New images can be posted to `/api/live`. 
For more details look at "app/resources.py" where all API-Endpoints are defined. In "camerahandler.py" is an example
script for receiving images via a socket and sending it to the API.   
In the Webinterface are following 
pages:  
- Galleries: Create and delete galleries. Organize images in galleries.
- Liveview: Show images as soon as they get processed. Empty on refresh
- Recent Classifications: Shows the recent classifications and gives possibility to confirm or correct the classifications. 
- Model Information: Shows information about the trained models. A new training process can be started from here.

The prototype classifies persons as "unknown" as soon as the maximum probability from the classifier is below a configurable threshold.

## TODO
This application is only a prototype and lacks security features, a useful architecture, usability, design and a 
thorough documentation. As such, the "only" todo is to rewrite the prototype as a full fledged application.
            
    