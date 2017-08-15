__OpenCV__  
change make -j8 to max number of parallel processes
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

__Change to corresponding virtualenv path__ 

```
cd ~/envs/keras_tf/lib/python2.7/site-packages/
ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so
```
__Install Requirements__  
```python -r requirements-txt```

**Run Postgres Docker**  
`sudo docker run --name postgres -e POSTGRES_PASSWORD=password -d -p 5432:5432 postgres`  
postgres available under `0.0.0.0:5432`

**Access Postgres Docker**  
`sudo docker exec -i -t <name> /bin/bash`

**Run Redis Docker** 
`sudo docker run -d --name redis -p 6379:6379 redis`  
redis available under `0.0.0.0:6379`

__Alternatively just install Redis locally__ 

**Configure Application**
- Copy `default_config_template.py` to `default_config.py`
- Adjust values inside `default_config.py`

**Initialize Database**  
`python manage.py db init`  
`python manage.py db migrate`  
`python manage.py db upgrade`


**Run Celery Worker**  
`celery worker -A celery_worker.celery --loglevel=info`

**Run Application**  
`python manage.py run`


**TODO**
- Backend
    - Dataset Loading. Change Path of unknown images or change loading code [DONE]
    - Training of Recognizer as celery background task [DONE]
    - Save label dictionary after training in database/somewhere [DONE]
    - Return useful information after classification [DONE]
    - Class-individual augmentation to target number of images [DONE]
    - Cross-validation for score of current model (background task) [DONE]
    - Save more info of model after training [DONE]
        - Number of images [DONE]
        - Average number of base images per class [DONE]
        - Number of images without a found face [DONE]
    - Investigate Memory Leaks [DONE][Solved by only allowing 1 Task per Worker process]
        - https://github.com/celery/celery/issues/3813
        - Celery tasks don't get closed after finishing
            - Maybe `del X,y` etc?
        - prevent multiple trainings tasks 
            - https://github.com/celery/celery/issues/3436
            - http://loose-bits.com/2010/10/distributed-task-locking-in-celery.html
    - Refactor App
        - Pull code out of \__init\__.py
    - Save last classifications (image and classification result (gallery)) [DONE]
    - Create socket.io endpoint for classification output [DONE]
        - Draw bounding box in image, with classification result [DONE]
        - Make annotation dependent on parameter [DONE]
    - Load Classifier + Labels on startup [DONE]
        - Include check if classifier exists before loading attempt... 
    - Check if gallery exists in database before loading label for classification (resync if not)
    - Security check for camera listener, don't allow images from everywhere...
    - Start camera listener socket on celery start [DONE][REMOVED!!] 
        - In API-Endpoint, check if task is running before attempting to start
        - Replace camera socket with external script and api endpoint for new images, should prevent celery task problems [DONE]
    - List of running tasks, their ids/urls and types (API-Endpoint) [DONE]
    - Replace TF/Keras Dependency with custom or independent library for image augmentation
    - Run FaceDetection on each subject and unknown image and save bounding box in database
- Frontend [DONE]
    - Index Page [STARTED]
    - Navigation Bar [STARTED]
    - Gallery overview [DONE]
        - Create new Gallery [DONE]
        - Show Gallery [DONE]
        - Move images  [DONE]
    - live classification view (hook up to socket.io endpoint for classification output) [DONE]
    - show last classifications [DONE]
        - mark classification as wrong [DONE]
            - is unknown -> to unknown [DONE]
            - is subject -> to subject gallery [DONE]
    - upload image and classify it
    - show running tasks [DONE] (Sort of)
    - show stats of current model[DONE]
        - retrain model [DONE]
    - Local hosting of js and css libraries
        - bootstrap
        - jquery
        - socket io
            
            
    