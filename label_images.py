import tensorflow as tf
import sys
import argparse
import os

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Path to the directory where images are stored", type=str)
    parser.add_argument("labels_file", help="Path to labels.txt", type=str)
    parser.add_argument("retrained_graph", help="Path to retrained_graph.pb", type=str)
    return parser.parse_args()

def list_directory(input_dir, ends_with=""):
    """ List files in a directory 
    Arguments:
        input_dir -- directory to read
        ends_with -- list only files whose filename end match "ends_with" 
                     pattern (optional)
    """
    files = []
    for f in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, f)) and \
                          f.endswith(ends_with):
            files.append(os.path.join(input_dir, f)) 
    return sorted(files)

# change this as you see fit
args = get_arguments()
image_path = args.image_dir


# Read in the image_data

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile(args.labels_file)]

# Unpersists graph from file
with tf.gfile.FastGFile(args.retrained_graph, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

image_paths = list_directory(args.image_dir, ends_with=".jpg")

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    for image_path in image_paths:
        print("Predicting: " + image_path)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        #for node_id in top_k:
        #    human_string = label_lines[node_id]
        #    score = predictions[0][node_id]
        #    print('%s (score = %.5f)' % (human_string, score))
        

        #filename = "results.txt"    
        with open(image_path + ".pred", 'w') as f:
            #f.write('\n**%s**\n' % (image_path))
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                f.write('%s %.5f\n' % (human_string, score))
    
    
