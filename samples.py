
import os
import itertools
import numpy as np
import cv2
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras import backend as K

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

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret

def main():

    samples = [os.path.join('./samples', fn) for fn in os.listdir('./samples')]

    #model = load_model('model.h5', custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
    with CustomObjectScope({'<lambda>': lambda y_true, y_pred: y_pred}):
        model = load_model('model.h5')
    #print(model.summary())

    input_data_ = model.get_layer('the_input').input
    y_pred_ = model.get_layer('softmax').output
    test_func = K.function([input_data_], [y_pred_])

    X_data = []

    for sample in samples:
        X_data.append(cv2.resize(cv2.imread(sample, cv2.IMREAD_COLOR if CHANNELS==3 else cv2.IMREAD_GRAYSCALE), (WIDTH, HEIGHT)).T.reshape((WIDTH, HEIGHT, CHANNELS)).astype(np.float32) / 255)

    X_data = np.array(X_data)

    print(X_data.shape)

    decoded_res = decode_batch(test_func, X_data)
    c = 0
    for i, r in enumerate(decoded_res):
        print(samples[i], r, (' !' if os.path.basename(samples[i]).split('.')[0] != r else ''))
        if os.path.basename(samples[i]).split('.')[0] != r:
            c += 1
    print(c)
    
if __name__ == '__main__':
    main()