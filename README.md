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


**TODO**  
- Dataset Loading. Change Path of unknown images or change loading code
- Replace TF/Keras Dependency with custom or independent library for image augmentation
- Run FaceDetection on each subject and unknown image and save bounding box in database