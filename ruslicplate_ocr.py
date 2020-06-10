# -*- coding: utf-8 -*-
'''
This requires editdistance packages:
pip install editdistance
'''
import os
import itertools
import codecs
import re
import datetime
import editdistance
import numpy as np
from scipy import ndimage
import pylab
import cv2
import random
import math
from PIL import ImageFont, ImageDraw, Image  
from string import ascii_letters, ascii_lowercase, ascii_uppercase, digits, punctuation, digits
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

def russian_strings():
    global russian_letters
    global russian_lowercase
    global russian_uppercase
    russian_lowercase = ''.join(chr(i) for i in range(1072, 1078)) + chr(1105) + ''.join(chr(i) for i in range(1078, 1104))
    russian_uppercase = ''.join(chr(i) for i in range(1040, 1046)) + chr(1025) + ''.join(chr(i) for i in range(1046, 1072))
    russian_letters = russian_lowercase + russian_uppercase

russian_strings()

OUTPUT_DIR = 'ruslicplate_ocr'

alphabet = u'0123456789ABCEHKMOPTXY'#u'0123456789ABCDEHKMOPTXY'

np.random.seed(55)

# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes grayscale with pixels ranging from 0 to 1

def speckle(img, maxseverity=1.0):
    severity = np.random.uniform(0, maxseverity)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck

def cv2_rounded_rectangle(img, topLeft, bottomRight, lineColor, thickness=1, lineType=cv2.LINE_8, shift= 0, cornerRadius = 0):
    #corners:
    #p1 - p2
    #|     |
    #p4 - p3

    p1 = topLeft
    p2 = bottomRight[0], topLeft[1]
    p3 = bottomRight
    p4 = topLeft[0], bottomRight[1]

    if thickness < 0:

        cv2.rectangle(img, (p1[0] + cornerRadius, p1[1]), (p2[0] - cornerRadius, p2[1] + cornerRadius), lineColor, thickness, lineType, shift);
        cv2.rectangle(img, (p1[0], p1[1] + cornerRadius), (p3[0], p3[1] - cornerRadius), lineColor, thickness, lineType, shift);
        cv2.rectangle(img, (p4[0] + cornerRadius, p4[1] - cornerRadius), (p3[0] - cornerRadius, p3[1]), lineColor, thickness, lineType, shift);

        cv2.ellipse(img, (p1[0] + cornerRadius, p1[1] + cornerRadius), (cornerRadius, cornerRadius), 180.0, 0, 90, lineColor, thickness, lineType, shift);
        cv2.ellipse(img, (p2[0] - cornerRadius, p2[1] + cornerRadius), (cornerRadius, cornerRadius), 270.0, 0, 90, lineColor, thickness, lineType, shift);
        cv2.ellipse(img, (p3[0] - cornerRadius, p3[1] - cornerRadius), (cornerRadius, cornerRadius), 0.0, 0, 90, lineColor, thickness, lineType, shift);
        cv2.ellipse(img, (p4[0] + cornerRadius, p4[1] - cornerRadius), (cornerRadius, cornerRadius), 90.0, 0, 90, lineColor, thickness, lineType, shift);

    else:
        cv2.line(img, (p1[0] + cornerRadius, p1[1]), (p2[0] - cornerRadius, p2[1]), lineColor, thickness, lineType, shift);
        cv2.line(img, (p2[0], p2[1] + cornerRadius), (p3[0], p3[1] - cornerRadius), lineColor, thickness, lineType, shift);
        cv2.line(img, (p4[0] + cornerRadius, p4[1]), (p3[0] - cornerRadius, p3[1]), lineColor, thickness, lineType, shift);
        cv2.line(img, (p1[0], p1[1] + cornerRadius), (p4[0], p4[1] - cornerRadius), lineColor, thickness, lineType, shift);

        cv2.ellipse(img, (p1[0] + cornerRadius, p1[1] + cornerRadius), (cornerRadius, cornerRadius), 180.0, 0, 90, lineColor, thickness, lineType, shift);
        cv2.ellipse(img, (p2[0] - cornerRadius, p2[1] + cornerRadius), (cornerRadius, cornerRadius), 270.0, 0, 90, lineColor, thickness, lineType, shift);
        cv2.ellipse(img, (p3[0] - cornerRadius, p3[1] - cornerRadius), (cornerRadius, cornerRadius), 0.0, 0, 90, lineColor, thickness, lineType, shift);
        cv2.ellipse(img, (p4[0] + cornerRadius, p4[1] - cornerRadius), (cornerRadius, cornerRadius), 90.0, 0, 90, lineColor, thickness, lineType, shift);

def random_perspective(img, Kw=0.25, Kh=0.25):
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    Dw0, Dw1 = random.randint(0, int(w * Kw)), random.randint(0, int(w * Kw))
    Dh0, Dh1 = random.randint(0, int(h * Kh)), random.randint(0, int(h * Kh))
    c = random.randint(0,3)
    if c == 0:
        pts2 = np.float32([[0 + Dw0, 0], [0, h], [w - Dw1, 0], [w, h]])
    elif c == 1:
        pts2 = np.float32([[0, 0], [0 + Dw0, h], [w, 0], [w - Dw1, h]])
    elif c == 2:
        pts2 = np.float32([[0, 0 + Dh0], [0, h - Dh1], [w, 0], [w, h]])
    else:
        pts2 = np.float32([[0, 0], [0, h], [w, 0 + Dh0], [w, h - Dh1]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img

def random_rotate(img, degree=8):
    h, w = img.shape[:2]
    ct = (int(w/2), int(h/2))
    angle = random.randint(-degree, degree)
    M = cv2.getRotationMatrix2D(ct, angle, 1)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img

def random_figures(sz, n=800, blur_kernelsize_factor=30):

    channels = 3

    img = np.zeros((sz[1], sz[0], channels), np.uint8)

    if random.randint(0, 1) == 1:
        img[::] = (255, 255, 255)

    height, width = img.shape[:2]

    Kf = 0.2

    for i in range(n):

        f = random.randint(0, 2)

        if f == 0:

            pt1 = (random.randint(0, width), random.randint(0, height))
            pt2 = (random.randint(0, width), random.randint(0, height))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            thickness = random.randint(1, 10)
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_8);

        elif f == 1:

            pt1 = (random.randint(0, width), random.randint(0, height))
            pt2 = (pt1[0] + random.randint(0, int(width * Kf)), pt1[1] + random.randint(0, int(height * Kf)))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            thickness = random.randint(-1, 0)
            cv2.rectangle(img, pt1, pt2, color, thickness, cv2.LINE_8);

        else:

            pt1 = (random.randint(0, width), random.randint(0, height))
            axes = (random.randint(0, int(width * Kf)), random.randint(0, int(height * Kf)))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            angle = random.randint(0, 360)
            startangle = random.randint(0, 360)
            endangle = random.randint(0, 360)
            thickness = random.randint(-1, 0)
            cv2.ellipse(img, pt1, axes, angle, startangle, endangle, color, thickness, cv2.LINE_8);

    k = random.randint(0, blur_kernelsize_factor) * 2 + 1
    img = cv2.GaussianBlur(img, (k, k), 0)
    if len(img.shape) < 3:
        img = img.reshape(img.shape[0], img.shape[1], 1)

    return img

def place_samples(samples, palette=((255, 255, 255, 0), (0, 0, 0))):

    #samples: [{'text': <string>, 'font': <string>, 'fontsize': <integer>, 'fontindex': <integer>, 'offset': (<integer>, <integer>), 'scale': (<float>, <float>), 'incline': <degrees float>, 'kdeltaheight': <factor float>}, ...]

    def get_Dh(font, Kh):
        if Kh != 1.0:
            _, h = font.getsize('A')
            return int(h * (Kh - 1))
        return 0

    def int1(f):
        return int(f) + 1

    fonts = [ImageFont.truetype(sample['font'], sample['fontsize'], sample['fontindex'] if 'fontindex' in sample else 0, encoding='') for sample in samples]

    sizes = []
    for i, sample in enumerate(samples, start=0):
        (w, h) = fonts[i].getsize(sample['text'])
        Dh = get_Dh(fonts[i], sample['kdeltaheight'] if 'kdeltaheight' in sample else 1.0)
        h = h + Dh
        if 'scale' in sample:
            (w, h) = (int1(w * sample['scale'][0]), int1(h * sample['scale'][1]))
        if 'incline' in sample:
            w += int1(h * math.sin(math.radians(abs(sample['incline']))))
        sizes.append((w, h))

    (width, height) = (0, 0)

    for i, size in enumerate(sizes, start=0):

        width += size[0] + ((samples[i]['offset'][0] if 'offset' in samples[i] else 0) if i != 0 else 0)

        if i != 0:

            dy = samples[i]['offset'][1] if 'offset' in samples[i] else 0
            h = size[1]

            y += dy

            if y < 0:
                height += -y
                y0 += -y
                y = 0

            if y + h > height:
                height = y + h

        else:
            height = size[1]
            y0 = 0
            y = 0

    img = Image.new('RGBA', (width, height), palette[0])

    for i, sample in enumerate(samples, start=0):
        offset = (offset[0] + (sample['offset'][0] if 'offset' in sample else 0) if i != 0 else 0,
                  offset[1] + (sample['offset'][1] if 'offset' in sample else 0) if i != 0 else y0)

        (w, h) = fonts[i].getsize(sample['text'])
        Dh = get_Dh(fonts[i], sample['kdeltaheight'] if 'kdeltaheight' in sample else 1.0)
        h = h + Dh
        patch = Image.new('RGBA', (w, h), (palette[0][0], palette[0][1], palette[0][2], 0))
        ImageDraw.Draw(patch).text((0, Dh), sample['text'], font=fonts[i], fill=palette[1])

        #correct for antialias
        #ImageDraw.Draw(patch).text((0, Dh), sample['text'], font=fonts[i], fill=palette[1])
        #patch = patch.resize((2 * w, 2 * h), Image.ANTIALIAS).resize((w, h), Image.ANTIALIAS)

        if 'incline' in sample or 'scale' in sample:

            if 'scale' in sample:
                (w, h) = (int1(w * sample['scale'][0]), int1(h * sample['scale'][1]))

            patch = patch.resize((w, h), Image.ANTIALIAS)

            if 'incline' in sample:

                dw = int1(h * math.sin(math.radians(abs(sample['incline']))))
                w += dw
                inclined = Image.new('RGBA', (w, h), (palette[0][0], palette[0][1], palette[0][2], 0))
                inclined.paste(patch, (dw if sample['incline'] > 0 else 0, 0), patch)

                mat = np.array(inclined)
                pts1 = np.float32([[0, h], [w, h], [w, 0]])
                pts2 = np.float32([[0, h], [w, h], [w + (-dw if sample['incline'] > 0 else dw), 0]])
                M = cv2.getAffineTransform(pts1, pts2)
                mat = cv2.warpAffine(mat, M, (w, h), flags=cv2.INTER_CUBIC)
                inclined = Image.fromarray(mat)

                patch = inclined

        img.paste(patch, offset, patch)
        offset = (offset[0] + sizes[i][0], offset[1])

    return img

def ruslicplateframe(size=None, frameoffset=None, framewidth=None, regionwidth=None, background=None, insidecolor=None, framecolor=None):
    channels = 3

    #GOST
    scale = (1.0 if size == None else size[0] / 520) if framewidth == None else framewidth / 520
    framesize = int(round(520 * scale)), int(round(112 * scale))
    boltoffset = int(round(20 * scale)), int(round(56 * scale))
    boltdiametr = int(round(7 * scale))
    h1 = int(round(58 * scale))
    h2 = int(round(76 * scale))
    h3 = int(round(58 * scale))
    h4 = int(round(20 * scale))

    if size == None:
       size = framesize

    if frameoffset == None:
        frameoffset = int(round(size[0] - framesize[0]) / 2), int(round(size[1] - framesize[1]))

    if background == None:
        img = random_figures(size, n = 500, blur_kernelsize_factor=10)
    else:
        img = np.zeros((size[1], size[0], channels), np.uint8)
        img[::] = background

    cornerradius = int(round(20 * scale))
    framethickness = int(round(6 * scale))
    dividerwidth = int(round(6 * scale))
    regionwidth = int(round(120 * scale)) if regionwidth == None else regionwidth
    mainwidth = framesize[0] - 2 * framethickness - regionwidth - dividerwidth

    if framecolor != None:
        cv2_rounded_rectangle(img, (frameoffset), (frameoffset[0] + framesize[0], frameoffset[1] + framesize[1]), framecolor, -1, cv2.LINE_AA, 0, cornerradius)
    if insidecolor != None:
        cv2_rounded_rectangle(img, (frameoffset[0] + framethickness, frameoffset[1] + framethickness), (frameoffset[0] + framethickness + mainwidth, frameoffset[1] + framesize[1] - framethickness), insidecolor, -1, cv2.LINE_AA, 0, cornerradius - framethickness)
        cv2_rounded_rectangle(img, (frameoffset[0] + framethickness + mainwidth + dividerwidth, frameoffset[1] + framethickness), (frameoffset[0] + framesize[0] - framethickness, frameoffset[1] + framesize[1] - framethickness), insidecolor, -1, cv2.LINE_AA, 0, cornerradius - framethickness)

    if framecolor != None:
        cv2.circle(img, (frameoffset[0] + boltoffset[0], frameoffset[1] + boltoffset[1]), int(round(boltdiametr / 2)), framecolor, -1, cv2.LINE_AA, 0)
        cv2.circle(img, (frameoffset[0] + framesize[0] - boltoffset[0], frameoffset[1] + boltoffset[1]), int(round(boltdiametr / 2)), framecolor, -1, cv2.LINE_AA, 0)

    t1 = int(round((framesize[1] - 2 * framethickness - h3 - h4) / 3))
    t2 = int(round(t1 / 2))

    r0 = ((frameoffset[0], frameoffset[1]), (frameoffset[0] + framesize[0], frameoffset[1] + framesize[1]))
    r1 = ((frameoffset[0] + boltoffset[0] + int(round(boltdiametr / 2)), frameoffset[1] + framethickness), (frameoffset[0] + framethickness + mainwidth, frameoffset[1] + framesize[1] - framethickness))
    r2 = ((frameoffset[0] + framethickness + mainwidth + dividerwidth, frameoffset[1] + framethickness), (frameoffset[0] + framesize[0] - boltoffset[0] - int(round(boltdiametr / 2)), frameoffset[1] + framethickness + t1 + h3 + t2))
    r3 = ((frameoffset[0] + framethickness + mainwidth + dividerwidth, frameoffset[1] + framethickness + t1 + h3 + t2), (frameoffset[0] + framesize[0] - framethickness, frameoffset[1] + framesize[1] - framethickness))

    return img, r0, r1, r2, r3, (h1, h2, h3, h4)

def ruslicplate(letters, size):

    framewidth = size[0]
    #framewidth = random.randint(int(size[0] * 0.6), size[0])
    #frameoffset = (random.randint(0, size[0] - framewidth), random.randint(0, size[1] - int(112/520 * framewidth)))
    regionwidth = (int(130 * framewidth/520) if len(letters) > 8 else int(110 * framewidth/520))
    regionwidth = random.randint(int((1 - 0.25) * regionwidth), int((1 + 0.25) * regionwidth))
    background=(255, 255, 255)
    insidecolor = random.randint(160, 255)
    insidecolor = (insidecolor, insidecolor, insidecolor)
    if random.randint(0, 3) == 0:
        framecolor = None
        fontcolor = random.randint(0, 96)
        fontcolor = (fontcolor, fontcolor, fontcolor)
    else:
        framecolor = random.randint(0, 96)
        framecolor = (framecolor, framecolor, framecolor)
        fontcolor = framecolor

    #frm, r0, r1, r2, r3, hs = ruslicplateframe(size=size, frameoffset=frameoffset, framewidth=framewidth, regionwidth=regionwidth, insidecolor=insidecolor, framecolor=framecolor)
    frm, r0, r1, r2, r3, hs = ruslicplateframe(size=size, regionwidth=regionwidth, background=background, insidecolor=insidecolor, framecolor=framecolor)
    font = 'RoadNumbers2.0.ttf'
    fontsize1 = 80
    fontsize2 = 80
    fontsize3 = 60
    kdeltaheight = 1.3

    if len(letters) != 0:

        samples = []
        samples.append({'text': letters[0], 'font': font, 'fontsize': fontsize1, 'kdeltaheight': kdeltaheight, 'offset': (0, 0)})
        samples.append({'text': letters[1], 'font': font, 'fontsize': fontsize2, 'kdeltaheight': kdeltaheight, 'offset': (5, 0)})
        samples.append({'text': letters[2], 'font': font, 'fontsize': fontsize2, 'kdeltaheight': kdeltaheight, 'offset': (5, 0)})
        samples.append({'text': letters[3], 'font': font, 'fontsize': fontsize2, 'kdeltaheight': kdeltaheight, 'offset': (5, 0)})
        samples.append({'text': letters[4], 'font': font, 'fontsize': fontsize1, 'kdeltaheight': kdeltaheight, 'offset': (5, 0)})
        samples.append({'text': letters[5], 'font': font, 'fontsize': fontsize1, 'kdeltaheight': kdeltaheight, 'offset': (0, 0)})
        img = place_samples(samples, palette=((insidecolor[0], insidecolor[1], insidecolor[2], 0), fontcolor))
        w, h = img.size
        back = Image.new('RGB', (w, h), insidecolor)
        back.paste(img, (0, 0), img)
        img = np.array(back)

        w = r1[1][0] - r1[0][0]
        h = r1[1][1] - r1[0][1]

        Dw2 = random.randint(int((0.025 - 0.02) * w), int((0.025 + 0.02) * w))
        Dw1 = 0#Dw2
        shift = 0#random.randint(-Dw1, Dw2)
        Dw1 += shift
        Dw2 -= shift
        Dh2 = random.randint(int((0.1 - 0.08) * h), int((0.1 + 0.08) * h))
        Dh1 = Dh2
        shift = random.randint(-Dh1, Dh2)
        Dh1 += shift
        Dh2 -= shift
        w -= Dw1 + Dw2
        h -= Dh1 + Dh2

        img = cv2.resize(img, (w, h))

        frm[r1[0][1]+Dh1:r1[1][1]-Dh2,r1[0][0]+Dw1:r1[1][0]-Dw2] = img

        samples = []
        samples.append({'text': letters[6], 'font': font, 'fontsize': fontsize3, 'kdeltaheight': kdeltaheight, 'offset': (0, 0)})
        samples.append({'text': letters[7], 'font': font, 'fontsize': fontsize3, 'kdeltaheight': kdeltaheight, 'offset': (0, 0)})
        if len(letters) > 8:
            samples.append({'text': letters[8], 'font': font, 'fontsize': fontsize3, 'kdeltaheight': kdeltaheight, 'offset': (0, 0)})
        img = place_samples(samples, palette=((insidecolor[0], insidecolor[1], insidecolor[2], 0), fontcolor))
        w, h = img.size
        back = Image.new('RGB', (w, h), insidecolor)
        back.paste(img, (0, 0), img)
        img = np.array(back)

        w = r2[1][0] - r2[0][0]
        h = r2[1][1] - r2[0][1]

        Dw2 = random.randint(int((0.05 - 0.03) * w), int((0.05 + 0.03) * w))
        Dw1 = Dw2
        shift = random.randint(-Dw1, Dw2)
        Dw1 += shift
        Dw2 -= shift
        Dh2 = random.randint(int((0.05 - 0.03) * h), int((0.05 + 0.03) * h))
        Dh1 = Dh2
        shift = random.randint(-Dh1, Dh2)
        Dh1 += shift
        Dh2 -= shift
        w -= Dw1 + Dw2
        h -= Dh1 + Dh2

        img = cv2.resize(img, (w, h))

        frm[r2[0][1]+Dh1:r2[1][1]-Dh2,r2[0][0]+Dw1:r2[1][0]-Dw2] = img

    w = r3[1][0] - r3[0][0]
    h = r3[1][1] - r3[0][1]

    Dw2 = random.randint(int((0.15 - 0.12) * w), int((0.15 + 0.12) * w))
    Dw1 = Dw2
    shift = random.randint(-Dw1, Dw2)
    Dw1 += shift
    Dw2 -= shift
    Dh2 = random.randint(int((0.1 - 0.08) * h), int((0.1 + 0.08) * h))
    Dh1 = Dh2
    shift = random.randint(-Dh1, Dh2)
    Dh1 += shift
    Dh2 -= shift
    w -= Dw1 + Dw2
    h -= Dh1 + Dh2

    img = random_figures((w, h), n = 25, blur_kernelsize_factor=5)

    frm[r3[0][1]+Dh1:r3[1][1]-Dh2,r3[0][0]+Dw1:r3[1][0]-Dw2] = img

    img = frm[r0[0][1]:r0[1][1],r0[0][0]:r0[1][0]]

    img = img.astype(np.float32) / 255
    img = speckle(img)
    img = (img * 255).astype(np.uint8)

    frm[r0[0][1]:r0[1][1],r0[0][0]:r0[1][0]] = img

    rx1 = r0[0][0] if random.random() < 0.2 else min(r1[0][0], r2[0][0])#rx1 = random.choice([r0[0][0], min(r1[0][0], r2[0][0])], weights=[0.2, 0.8]) #python 3.6
    ry1 = random.choice([r0[0][1], min(r1[0][1], r2[0][1])])
    rx2 = random.choice([r0[1][0], max(r1[1][0], r2[1][0])])
    ry2 = random.choice([r0[1][1], max(r1[1][1], r2[1][1])])
    rr = ((rx1, ry1), (rx2, ry2))

    frm = frm[rr[0][1]:rr[1][1],rr[0][0]:rr[1][0]]
    frm = cv2.resize(frm, size)

    frm = random_perspective(frm, Kw=0.08, Kh=0.025)
    frm = random_rotate(frm, degree=4)

    blur_kernelsize_factor = 10
    k = random.randint(0, blur_kernelsize_factor) * 2 + 1
    frm = cv2.GaussianBlur(frm, (k, k), 0)

    return frm

def generate_ruslicplate():
    chars = 'ABCEHKMOPTXY'#'ABCDEHKMOPTXY'
    string = ''
    string += ''.join(random.choice(chars) for i in range(1))
    string += ''.join(random.choice(digits) for i in range(3))
    string += ''.join(random.choice(chars) for i in range(2))
    string += ''.join(random.choice(digits) for i in range(2 if random.randint(0, 1) == 0 else 3))
    return string

# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise

def paint_text(text, w, h):
    img = ruslicplate(text, (256, 55))
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255
    img = np.expand_dims(img, 0)
    return img 


def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret


# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, minibatch_size,
                 img_w, img_h, downsample_factor, val_split,
                 absolute_max_string_len=9):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len

    def get_output_size(self):
        return len(alphabet) + 1

    def build_word_list(self, num_words):
        assert num_words % self.minibatch_size == 0
        assert (self.val_split * num_words) % self.minibatch_size == 0
        self.num_words = num_words
        self.cur_val_index = self.val_split
        self.cur_train_index = 0

    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 1])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if train and i > size - 4:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func('',)[0, :, :].T
                labels[i, 0] = self.blank_label
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = 1
                source_str.append('')
            else:
                word = generate_ruslicplate()
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = (
                        self.paint_func(word)[0, :, :].T)
                else:
                    X_data[i, 0:self.img_w, :, 0] = (
                        self.paint_func(word)[0, :, :].T)
                ydata = np.ones([self.absolute_max_string_len]) * -1
                ydata[0:len(word)] = text_to_labels(word)
                labels[i, :] = ydata
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = len(word)
                source_str.append(word)
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index,
                                 self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index,
                                 self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_train_begin(self, logs={}):
        self.build_word_list(16000)
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h)

    def on_epoch_begin(self, epoch, logs={}):
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h)

# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret


class VizCallback(keras.callbacks.Callback):

    def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, run_name)
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = decode_batch(self.test_func,
                                       word_batch['the_input'][0:num_proc])
            for j in range(num_proc):
                edit_dist = editdistance.eval(decoded_res[j],
                                              word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance:'
              '%.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        self.show_edit_distance(256)
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func,
                           word_batch['the_input'][0:self.num_display_words])
        if word_batch['the_input'][0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        for i in range(self.num_display_words):
            pylab.subplot(self.num_display_words // cols, cols, i + 1)
            if K.image_data_format() == 'channels_first':
                the_input = word_batch['the_input'][i, 0, :, :]
            else:
                the_input = word_batch['the_input'][i, :, :, 0]
            pylab.imshow(the_input.T, cmap='Greys_r')
            pylab.xlabel(
                'Truth = \'%s\'\nDecoded = \'%s\'' %
                (word_batch['source_str'][i], res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        pylab.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch + 1)))
        pylab.close()


def train(run_name, stop_epoch, img_w=297, img_h=64):
    # Input Parameters
    words_per_epoch = 16000
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))

    pool_size = 2
    minibatch_size = 32

    img_gen = TextImageGenerator(
        minibatch_size=minibatch_size,
        img_w=img_w,
        img_h=img_h,
        downsample_factor=(pool_size ** 2),
        val_split=words_per_epoch - val_words)

    start_epoch = 0

    if os.path.exists(os.path.join(OUTPUT_DIR, run_name)):
        files = os.listdir(os.path.join(OUTPUT_DIR, run_name))
        for f in files:
            if len(f.split('.')) < 2:
                continue
            if f.split('.')[-1] != 'h5':
                continue
            n = int(f.split('-')[1])
            if n > start_epoch:
                start_epoch = n
                modelfile = f

    if start_epoch == 0:

        # Network parameters
        conv_filters = 16
        kernel_size = (3, 3)
        time_dense_size = 32
        rnn_size = 512

        if K.image_data_format() == 'channels_first':
            input_shape = (1, img_w, img_h)
        else:
            input_shape = (img_w, img_h, 1)

        act = 'relu'
        input_data = Input(name='the_input', shape=input_shape, dtype='float32')
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

        conv_to_rnn_dims = (img_w // (pool_size ** 2),
                            (img_h // (pool_size ** 2)) * conv_filters)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        # cuts down input size going into RNN:
        inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

        # Two layers of bidirectional GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(rnn_size, return_sequences=True,
                    kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(rnn_size, return_sequences=True,
                     go_backwards=True, kernel_initializer='he_normal',
                     name='gru1_b')(inner)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(rnn_size, return_sequences=True,
                    kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                     kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

        # transforms RNN output to character activations:
        inner = Dense(img_gen.get_output_size(), kernel_initializer='he_normal',
                      name='dense2')(concatenate([gru_2, gru_2b]))
        y_pred = Activation('softmax', name='softmax')(inner)
        Model(inputs=input_data, outputs=y_pred).summary()

        labels = Input(name='the_labels',
                       shape=[img_gen.absolute_max_string_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(
            ctc_lambda_func, output_shape=(1,),
            name='ctc')([y_pred, labels, input_length, label_length])

        # clipnorm seems to speeds up convergence
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    else:
        with CustomObjectScope({'<lambda>': lambda y_true, y_pred: y_pred}):
            model = load_model(os.path.join(OUTPUT_DIR, run_name, modelfile))
        input_data = model.get_layer('the_input').input
        y_pred = model.get_layer('softmax').output

    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    viz_cb = VizCallback(run_name, test_func, img_gen.next_val())

    checkpoint = ModelCheckpoint(filepath=os.path.join(OUTPUT_DIR, run_name, 'model-{epoch:03d}-{val_loss:.3f}.h5'), monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

    tensorboard = TensorBoard(log_dir=os.path.join(OUTPUT_DIR, run_name, 'logs'), batch_size=minibatch_size)

    model.fit_generator(
        generator=img_gen.next_train(),
        steps_per_epoch=(words_per_epoch - val_words) // minibatch_size,
        epochs=stop_epoch,
        validation_data=img_gen.next_val(),
        validation_steps=val_words // minibatch_size,
        callbacks=[viz_cb, img_gen, checkpoint, tensorboard],
        initial_epoch=start_epoch)


if __name__ == '__main__':
    #run_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_name = 'train'
    train(run_name, 150, 128, 28)

    #for i in range(102):
    #    text = generate_ruslicplate()
    #    cv2.imwrite('./samples1/' + text + '.jpg', cv2.cvtColor(cv2.resize(ruslicplate(text, (256, 55)), (128, 28)), cv2.COLOR_BGR2GRAY))
