#!/usr/bin/python
from asyncore import close_all

import dsift, sift
import os
from numpy import *
from pylab import *
from PIL import Image
import cv2


PGM = 1


def hog_histogram(img_list, class_label, out_file, norm, debug):

    f = open(img_list)
    filenames = f.read().splitlines()
    f.close()

    image_vector = []
    ctr = 0

    count = 0
    end_count = len(filenames)

    for entry in filenames:

       #want

       dsift.process_image_dsift(entry, 'tmp.sift', 20, 10, resize=(200,200))
       l,d = sift.read_features_from_file('tmp.sift')

       # if debug:
       #    cv2.destroyAllWindows()
       #
       #    #used to draw on original image instead of resized image
       #    im = array(Image.open(entry))
       #    tmp_im = cv2.resize(im, (200,200))
       #    tmp_im = cv2.cvtColor(tmp_im, cv2.COLOR_BGR2GRAY)
       #    sift.plot_features(tmp_im, l, True)
       #
       #    show()

       image_vector.append(d.flatten().tolist())
       os.remove('tmp.sift')
       os.remove('tmp.frame')

       if not PGM:
          os.remove('tmp.pgm')

       if out_file != None:
          f = open(out_file, 'a')

          if class_label == 1:
             vector = '+1'
          else:
             vector = '-1'

          vec_length = len(image_vector[ctr])

          for i, dim in zip(xrange(vec_length), xrange(1, vec_length+1)):
             vector += ' ' + str(dim) + ':' + str(image_vector[ctr][i])

          f.write(vector)
          f.write('\n')

          f.close()

       ctr += 1

       count += 1
       #sys.stdout.write('\r%i/%i' % (count, end_count))
       #sys.stdout.flush()

    return image_vector

if __name__ == '__main__':

     debug = 1

     img_list = sys.argv[1]
     out_file = sys.argv[2]
     class_label = int(sys.argv[3])
     norm = -1

     hog_histogram(img_list, class_label, out_file, norm, debug=True)
