import cv2
import os
import numpy as np
import time

from face_utils import *
from graphic_utils import *


def reset_files():
    """helper function, want to make sure we analyze one picture at a time """
    if os.path.isfile('image_list.txt'):
        os.remove('image_list.txt')
    if os.path.isfile('output'):
        os.remove('output')


def live(seconds=-1, scalefactor=1.0, screen_loc = (0,0)):
    """
    main pipeline connecting face detection, face normalization, score prediction, and output
    args:   seconds: duration of demo, -1 for continuous
            scalefactor: scale size of final display
            screen_loc: location of display on screen
    """

    # initialize values
    cap = cv2.VideoCapture(0)

    # to add more traits
    #   add trait name below to list, 'traits'
    #   add trait name and model file location to dictionary, 'models', in face_utils.py

    # the order of this list is the order the traits will appear when pressing 'n' in the demo
    traits = [
        'Trustworthiness',
        'Dominance'
    ]
    current_trait = 0

    graphic = Graphics()
    background = cv2.imread('background.jpg')

    debug = False
    begin_time = time.time()

    # loop through frames
    while True:

        _, frame = cap.read()

        background_height, background_height_top = int(frame.shape[0]/5), 100
        background = cv2.resize(background, (frame.shape[1], background_height))
        background_top = cv2.resize(background, (frame.shape[1], background_height_top))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)
        face = None

        # find largest face in image, create instance of Face class
        ret = find_largest_face(gray)
        if ret:
            face_coords, face_splice = ret
            face = Face(face_coords, face_splice, traits[current_trait])

        # find eyes and normalize image of face
        # predict trait score from zero to one
            if face.find_eyes() and face.normalize() and face.predict_trait_value():
                if debug:
                    # in top left corner, display the image that will actually be processed
                    frame[0:200, 0:200] = cv2.cvtColor(cv2.resize(face.norm_image, (200, 200)), cv2.COLOR_GRAY2BGR)

        # add graphics' backgrounds at top and bottom of image
        frame = np.concatenate((background_top, frame), axis=0)
        frame = np.concatenate((frame, background), axis=0)

        # create graphic on image
        if face is None:
            p_arg = -1
        else:
            p_arg = round(face.p_val, 2) if (face.p_val > 0) else -1
        graphic.draw_graphics(frame, p_arg, traits[current_trait], background_height)

        # scale and show image
        frame = cv2.resize(frame, None, fx=scalefactor, fy=scalefactor, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Demo', frame)
        cv2.moveWindow('Demo', *screen_loc)

        # reset files, variables
        reset_files()
        del face

        # check duration
        if seconds != -1 and (time.time() - begin_time) >= seconds:
            cv2.destroyAllWindows()
            break

        # accept commands
        # 'r':reset, 'd':debug, 'n':next trait, 1-9:graph speed, 1-9:quit
        keypress = cv2.waitKey(30)
        if keypress == ord('r'):
            graphic.coord_list, graphic.max_p_val = [], 0
        elif keypress == ord('d'):
            debug = fabs(debug-1)
        elif keypress == ord('n'):
            current_trait = 0 if (current_trait == len(traits)-1) else (current_trait+1)
        elif keypress in range(49, 58):
            graphic.update_speed = (keypress - 48) * 3
        elif keypress == 27:
            cv2.destroyAllWindows()
            if os.path.exists('tmp.jpg'):
                os.remove('tmp.jpg')
            if os.path.exists('tmp.pgm'):
                os.remove('tmp.pgm')
            break


if __name__ == '__main__':
    live(seconds=-1, scalefactor=1, screen_loc=(0,0))
