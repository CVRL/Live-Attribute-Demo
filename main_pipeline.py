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
    if os.path.exists('tmp.jpg'):
        os.remove('tmp.jpg')
    if os.path.exists('tmp.pgm'):
        os.remove('tmp.pgm')

def process_media(current_trait=0, media_type=1, seconds=-1, multi_faces=False, infile=None, outfile=None, fps=-1, scalefactor=1.0, screen_loc = (0,0), debug=False):
    """
    main pipeline connecting face detection, face normalization, score prediction, and output
    args:   current_trait:
                number corresponding to trait in list, 'traits', i.e. 0 is Trustworthiness
            media_type:
                1 for webcam,
                2 for pre recorded video,
                3 for still picture
            seconds:
                <0 (i.e. -1) for a still image, a continuous demo, or all of a video
                0 to save single webcam frame or video frame
                >0 (i.e. 1) for desired duration of outupt video, or desired duration of webcam demo in seconds
            multi_faces:
                False to only analyze largest face
                True to analyze all potential faces with valid eyes
            infile:
                name of input file for prerecorded video
                name of input file for single image
            outfile:
                name of output file for single image
                name of output file for video
                name of output file for single webcam frame
            fps:
                frames per second of output video for video
            scalefactor:
                scale size of final display (does not affect scale of outfile)
            screen_loc:
                location of display on screen (does not affect scale of outfile)
    """
    # check argument validity
    if media_type not in [1, 2, 3]:
        raise ValueError('Improper media_type value. Set to 1 for webcam, 2 for video, 3 for still image')
    if media_type in [2, 3]:
        if not os.path.isfile(infile):
            raise IOError('Could not find input file: ' + infile)

    # initialize values
    # webcam
    if media_type == 1:
        cap = cv2.VideoCapture(0)

    # pre recorded video
    # make sure to use the proper video file for your system
    elif media_type == 2:
        cap = cv2.VideoCapture(infile)
        cap_width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        cap_height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        if not multi_faces:
            dim = (cap_width, cap_height + 100 + int(cap_height/5))
        else:
            dim = (cap_width, cap_height)
        fourCC = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
        videoFile = cv2.VideoWriter()
        videoFile.open(outfile, fourCC, fps, dim, True)

    # to add more traits
    #   add trait name below to list, 'traits'
    #   add trait name and model file location to dictionary, 'models', in face_utils.py

    # the order of this list is the order the traits will appear when pressing 'n' in the demo
    traits = [
        'Trustworthiness',
        'Dominance'
    ]

    graphic = Graphics()
    begin_time = time.time()

    # loop through frames
    while media_type == 3 or cap.isOpened():

        if media_type == 1 or media_type == 2:
            _, frame = cap.read()
        else:
            frame = cv2.imread(infile)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)
        face = None

        if multi_faces:  # find all faces in frame
            face_list = find_faces(gray)
        else:  # find largest face in image
            face_list = [find_largest_face(gray)]

        if face_list is not None:
            for face_coords in face_list:
                if face_coords is not None:
                    x, y, w, h = face_coords
                    face = Face(face_coords, gray[y:y+h, x:x+w], traits[current_trait])

                    # find eyes and normalize image of face
                    # predict trait score from zero to one
                    if face.find_eyes() and face.normalize() and face.predict_trait_value():
                        if multi_faces:
                            graphic.draw_graphis_multi_faces(frame, traits[current_trait], face.p_val, face_coords, debug)
                            reset_files()
                            del face
                            face = None
                        else:
                            if debug:
                                # in top left corner, display the image that will actually be processed
                                frame[0:200, 0:200] = cv2.cvtColor(cv2.resize(face.norm_image, (200, 200)), cv2.COLOR_GRAY2BGR)

        if not multi_faces:
            # set graphic args and create graphic on image
            if face is None:
                p_arg = -1
            else:
                p_arg = round(face.p_val, 2) if (face.p_val > 0) else -1

            media_arg = 3 if seconds == 0 else media_type

            frame = graphic.draw_graphics_single_face(media_arg, frame, p_arg, traits[current_trait])
        else:
            graphic.update_values(1)

        # save frame if necessary
        if media_type == 1 and seconds == 0:
            cv2.imwrite(outfile, frame)
        elif media_type == 2:
            if seconds == 0:
                cv2.imwrite(outfile, frame)
            else:
                videoFile.write(frame)
        elif media_type == 3:
            cv2.imwrite(outfile, frame)

        # scale and show image
        frame = cv2.resize(frame, None, fx=scalefactor, fy=scalefactor, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Demo', frame)
        cv2.moveWindow('Demo', *screen_loc)

        # reset files, variables
        reset_files()
        del face

        # check duration
        if media_type == 1 and seconds >= 0 and (time.time() - begin_time) >= seconds:
            reset_files()
            cv2.destroyAllWindows()
            break
        elif media_type == 2 and seconds >= 0 and (graphic.total_frames/fps) >= seconds:
            reset_files()
            cv2.destroyAllWindows()
            break
        elif media_type == 3:
            reset_files()
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
            reset_files()
            cv2.destroyAllWindows()
            break

    if media_type == 2:
        videoFile.release()
        videoFile = None


if __name__ == '__main__':
    process_media(
        current_trait=0,
        media_type=1,
        seconds=-1,
        multi_faces=True,
        debug=False,
        infile='still.jpg',
        outfile='out.jpg',
        fps=24,
        scalefactor=1,
        screen_loc=(0, 0)
    )

