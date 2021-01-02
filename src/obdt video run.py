import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import tkinter as tk
import threading
import time

from tkinter import filedialog
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


# ## Env setup

# In[58]:


# This is needed to display the images.
#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Object detection imports
# Here are the imports from the object detection module.

# In[59]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[60]:


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


# ## Download Model

# In[61]:


#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())




# ## Load a (frozen) Tensorflow model into memory.

graph = tf.Graph()
with graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[63]:


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# ## Helper code

# In[64]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  


# # Detection

# In[65]:


##Since we are using only CPU hence we need to convert video into frames for detection
fratef=20
def frameconv(pathIn, pathOut):
    if not os.path.exists(pathOut):  
      os.makedirs(pathOut)
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*50))    # added this line
      #roughly the multiplication fator is amount of time in milliseconds through which frame would be taken
      success,image = vidcap.read()
      countx=count+1
      print ('Read a new frame: ', countx, success)
      cv2.imwrite( pathOut + "\\frame%d.jpg" % countx, image)     # save frame as JPEG file
      count = count + 1

    count=count-1
    return count
PATH_TO_TEST_IMAGES_DIR = 'C:/Users/Avi Agrawal/Desktop'
pathvid=sys.argv[1]
print(pathvid)
amtframes=frameconv('.\\'+pathvid,'Frameconverts')
TEST_IMAGE_PATHS=[]
for a in range(1,amtframes+1):
  TEST_IMAGE_PATHS.append('Frameconverts/frame'+str(a)+'.jpg')
s=1
def dispth2():
  global TEST_IMAGE_PATHS
  global s
  s=len(TEST_IMAGE_PATHS)
  if s%2==0:
    s=(s/2)+1
  else:
    s=((s+1)/2)
  TEST_IMAGE_PATHS=TEST_IMAGE_PATHS[:int(s)]
  argm=''
  print(str(s)+' '+str(amtframes))
  os.system('python object_detection_thread2.py '+str(int(s))+' '+str(amtframes))
  

  
def thread2():
  dispth2()
  pass
t1 = threading.Thread(target=thread2, args=[])
t1.start()





# Size, in inches, of the output images.
IMAGE_SIZE = (24,16)
x=1

with graph.as_default():
    sess = tf.Session()
    for image_path in TEST_IMAGE_PATHS:
      test = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image = load_image_into_numpy_array(test)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image, axis=0)
      # Actual detection.  image_np_expanded = np.expand_dims(image, axis=0) 
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
      image,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
      image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      if x==1:
        frame = image_converted
        height, width, layers = frame.shape
        video = cv2.VideoWriter('Server Saved\\f1.mp4', 0, fratef, (width,height))
      video.write(image_converted)
      print('converted image ' + str(x))
      x=x+1
    

  
def combvideo():
  TEST_IMAGE_PATHS=[]
  for a in range(int(s),amtframes+1):
    TEST_IMAGE_PATHS.append('SavedObjects/'+str(a)+'.jpg')  
  fn=int(s)
  for image in TEST_IMAGE_PATHS:
    video.write(cv2.imread(image))
    print('Imported Frame '+str(fn))
    fn=fn+1
    video.write(cv2.imread(image))
  video.release()

t1.join()
combvideo()
video.release()

# In[ ]:




