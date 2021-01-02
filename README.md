# Object-Detection
Object Detection Using Tensorflow RCNN model. The model is trained on the COCO dataset which contains 90 objects. The detection rate is about 0.25 seconds for 600 x 400 images. The objects are also bounded into boxes. Minimum confidence level required will be `60%`  

Also deployed to be controlled via flask based web framework. Can detect objects from images, vidoes and live webcam. The detection peroformance was enhanced for videos by employing multithreading to perform alternate frame detection, giving peroformance increaments by about `42%`

## Requirements
Tensorflow  
OpenCV  
Pillow  
Flask  

## Usage  
Clone this repo, get the model and run mainserver.py.  
You can also use the webcam or video file directly.
