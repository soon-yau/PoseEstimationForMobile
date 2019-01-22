import tensorflow as tf
import numpy as np
import json
import argparse
import cv2
import os
import math
import time
import glob

def infer(frozen_pb_path, output_node_name, img_path, output_path=None):
    with tf.gfile.GFile(frozen_pb_path, "rb") as f:
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

    res = {}
    use_times = []
    with tf.Session() as sess:
        # if directory, then glob all files
        # if file, then do once
        for i in range(1):
            ori_img = cv2.imread(img_path)
            #print(ori_img)
            shape = input_image.get_shape().as_list()
            inp_img = cv2.resize(ori_img, (shape[1], shape[2]))
            st = time.time()
            heat = sess.run(output_heat, feed_dict={input_image: [inp_img]})
            infer_time = 1000 * (time.time() - st)
            #print("img_id = %d, cost_time = %.2f ms" % (img_id, infer_time))
            use_times.append(infer_time)
            print(heat.shape)

            grey_heat = np.squeeze(np.amax(heat, axis=3))
            max_val = np.max(grey_heat)
            print("max:", np.min(grey_heat), np.max(grey_heat))
            cv2.imwrite(output_path,255*grey_heat/max_val)
            #res[img_id] = np.squeeze(heat)
    print("Average inference time = %.2f ms" % np.mean(use_times))
    #return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_pb_path", type=str, default="")
    parser.add_argument("--img_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="pred_heatmap.jpg")
    parser.add_argument("--output_node_name", type=str, default='Convolutional_Pose_Machine/stage_5_out')
    parser.add_argument("--gpus", type=str, default="1")
    args = parser.parse_args()

    infer(args.frozen_pb_path, args.output_node_name, args.img_path, args.output_path)

