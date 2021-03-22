# Yolov5 jetson CSI camera video inference

## Features

* Multi-threading yolov5 inference with Nvidia jetson on-board camera (RaspberryPi camera mudule ver.2 is used).
* Save detection results to video file.

## Prerequisites and build
This work is derivated form [yolov5-multi-video-inference](https://github.com/edge-ai4cv/yolov5-multi-video-inference.git). So you can use that git repo as reference.
## Run "yolov5-multi-video"
```
./yolov5-csi-camera -csi [engine]
```
To interrup program, press "Esc" and  you can then access the saved video files. 

##Acknowledgments
* [wang-xinyu/tensorrt/yolov5](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5) for yolov5 tensorrt implementation.

* [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) for data structure to pass objects between threads.

