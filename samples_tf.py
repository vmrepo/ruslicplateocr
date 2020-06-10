
import os
import itertools
import numpy as np
import cv2
import tensorflow as tf

WIDTH = 128
HEIGHT = 28
CHANNELS = 1

alphabet = u'0123456789ABCEHKMOPTXY'#u'0123456789ABCDEHKMOPTXY'

def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret

def main():

    samples = [os.path.join('./samples', fn) for fn in os.listdir('./samples')]

    X_data = []

    for sample in samples:
        X_data.append(cv2.resize(cv2.imread(sample, cv2.IMREAD_COLOR if CHANNELS==3 else cv2.IMREAD_GRAYSCALE), (WIDTH, HEIGHT)).T.reshape((WIDTH, HEIGHT, CHANNELS)).astype(np.float32) / 255)

    X_data = np.array(X_data)

    print(X_data.shape)

    with tf.gfile.FastGFile('graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('softmax/truediv:0')
        print(softmax_tensor)

        input_tensor = sess.graph.get_tensor_by_name('the_input:0')
        print(input_tensor)

        predictions = sess.run(softmax_tensor, {'the_input:0': X_data})
        print(predictions.shape)
        decoded_res = decode_batch(predictions)
        c = 0
        for i, r in enumerate(decoded_res):
            print(samples[i], r, (' !' if os.path.basename(samples[i]).split('.')[0] != r else ''))
            if os.path.basename(samples[i]).split('.')[0] != r:
                c += 1
        print(c)

if __name__ == '__main__':
    main()
