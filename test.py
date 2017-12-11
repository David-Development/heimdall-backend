#!/usr/bin/env python

import sys
import redis
import concurrent.futures
import cv2
from heimdall.mqtt_handler import MqttHandler
import time

r = redis.StrictRedis(host="redis", port="6379", charset="utf-8", decode_responses=True)
fred = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def check_server():
    try:
        r.info()
    except redis.exceptions.ConnectionError:
        print("Error: cannot connect to redis server. Is the server running?")
        sys.exit(1)


def testVideoCamera():
    cap = cv2.VideoCapture('http://192.168.10.48:8080/cam.mjpg')

    counter = 0
    while counter < 100:  # read 100 frames
        ret, frame = cap.read()
        print(ret)
        # cv2.imshow('Video', frame)
        # if cv2.waitKey(1) == 27:
        #    exit(0)

        counter += 1

    executor.shutdown()


def testThreadPool():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    for num in fred:
        executor.submit(f, num)

def f(x):
    res = x * x
    r.rpush("test", res)


def testMQTT():
    mqttHandler = MqttHandler()
    mqttHandler.connect()

    for i in range(10):
        time.sleep(1)

    mqttHandler.disconnect()


def main():

    # Test MQTT
    testMQTT()

    #testThreadPool()

    #testVideoCamera()

    # print(r.lrange("test", 0, -1))


####################

if __name__ == "__main__":
    #check_server()
    #r.delete("test")
    main()
