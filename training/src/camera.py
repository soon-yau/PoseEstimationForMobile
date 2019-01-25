import numpy as np
import cv2
import tensorflow as tf
import numpy as np
import json
import argparse
import cv2
import os
import glob
import math
import time
import glob
from scipy.ndimage.filters import gaussian_filter

def detect_peak(heatmap):

    _, h, w, n_features = heatmap.shape
    colour_heat = np.zeros((h, w, 3), dtype=np.float32)

    # for each heat map, detect peak and draw circle
    threshold = 0.0
    for i in range(n_features):
        heat = heatmap[0,:,:,i]
        heat = gaussian_filter(heat, sigma=5)
        peak_coord = np.unravel_index(np.argmax(heat),heat.shape)
        peak = heat[peak_coord]
        # left foot is red
        # right foot is yellow
        colour = (255, 255, 255) if i<7 else (0, 0, 255)
        
        if peak > threshold:
            cv2.circle(colour_heat, peak_coord, 1, colour, -1)
            '''
            colour_heat = cv2.putText(colour_heat, 
                            text="%d %.1f"%(i,peak),
                            org=(peak_coord[0]-5, peak_coord[1]), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.2,
                            color=(255, 255, 255))
            '''

    return colour_heat


output_node_name = 'Convolutional_Pose_Machine/stage_5_out'
parser = argparse.ArgumentParser()
parser.add_argument("--frozen_pb_path", type=str, default="")
args = parser.parse_args()


with tf.gfile.GFile(args.frozen_pb_path, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

tf.import_graph_def(
    restored_graph_def,
    input_map=None,
    return_elements=None,
    name=""
)

graph = tf.get_default_graph()
input_image = graph.get_tensor_by_name("image:0")
output_heat = graph.get_tensor_by_name("%s:0" % output_node_name)


# open camera
cap = cv2.VideoCapture(1)
with tf.Session() as sess:
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        ori_shape = frame.shape
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        shape = input_image.get_shape().as_list()
        inp_img = cv2.resize(frame, (shape[1], shape[2]))
        heat = sess.run(output_heat, feed_dict={input_image: [inp_img]})

        grey_heat = 255*np.squeeze(np.amax(heat, axis=3))

        grey_heat = cv2.resize(grey_heat, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_AREA)
        color_heat = np.zeros((ori_shape[0], ori_shape[1], 3), dtype=np.float32)
        color_heat[:,:,2] = grey_heat
        

        #cv2.imwrite(output_path, grey_heat)
        merged_img = cv2.addWeighted(frame, 1.0, color_heat.astype(np.uint8), 1.0, 0)
        # add keypoint
        
        color_keypoints = detect_peak(heat)
        color_keypoints = cv2.resize(color_keypoints, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_AREA)
        merged_img = cv2.addWeighted(merged_img, 1.0, np.uint8(255*color_keypoints), 1.0, 0)
        
        # Display the resulting frame
        cv2.imshow('frame', merged_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
