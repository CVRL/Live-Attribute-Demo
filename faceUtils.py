import cv2
import sys
import os
from itertools import combinations

sys.path.insert(0, './liblinear/python/')
from liblinearutil import predict
from liblinearutil import svm_read_problem
from liblinearutil import load_model

from genAttrFeaturesHOG import hog_histogram
from math_utils import *


faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
traits = [
    'Trustworthiness',
    'Dominance'
]
models = {
    'Dominance':'models/hog_dom_train_6200.model',
    'Trustworthiness':'models/hog_trust_train_6200.model'
}


def findFaces(gray):
    """
    Uses Viola-Jones to find face with greatest area
    Args: gray - grayscale of image to be searched
    Return: coordinates of the largest face in the grayscale image
            splice of grascale image containing face
    """
    all_faces = faceCascade.detectMultiScale(
        image=gray,
        scaleFactor=1.2,
        minNeighbors=2,
        #minSize=(45,45)
    )
    if len(all_faces) == 0:
        return None
    return all_faces


def findLargestFace(gray):
    all_faces = findFaces(gray)
    if all_faces is None:
        return None
    else:
        areas = [areaOf(face) for face in all_faces]
        face_coords = all_faces[areas.index(max(areas))]
    return face_coords


def splice(image, coords):
    x, y, w, h = coords
    return image[y:y+h, x:x+w]


class Face:
    """ General class containing methods to be applied on detected faces """

    def __init__(self, image, coords, trait):
        self.x, self.y, self.w, self.h = coords
        self.splice = splice(image, coords)
        self.height, self.width = self.splice.shape[0], self.splice.shape[1]
        self.trait = traits[trait]

        self.eyes_coords = []
        self.eye_midpoint = ()
        self.eye_angle = 31

        self.norm_image = []

        self.p_val = -1

    def findEyes(self):
        """
        Detects eyes within the face, updates class values
        Args:
        Return: True if successful, None otherwise
        """
        # restrictions on eyes to maximize accuracy:
        #   within face
        #   angle between is less than or equal to 30 degrees
        #   least angle of all pairs
        #   both eyes in top half of face
        #   each eye on opposite half of face

        all_eyes = eyeCascade.detectMultiScale(
            image=self.splice,
            scaleFactor=1.1,
            minNeighbors=1
        )

        if len(all_eyes) == 0:
            return None

        # make list of eye centers in top half of the face
        eye_centers = []
        [eye_centers.append(centerOf(eye)) for eye in all_eyes if eye[1] < self.height/2]

        # find the combination of eyes at the least angle and on opposite sides of the face
        for (x1, y1), (x2, y2) in combinations(eye_centers, 2):
            # get left and right eye
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            # make sure they're on opposite halves of face
            if x1 < self.width/2 and x2 > self.width/2:
                # calculate angle between them
                angle = angleBetween((x1, y1, x2, y2))
                # update values if at a smaller angle
                if fabs(angle) < fabs(self.eye_angle):
                    self.eyes_coords, self.eye_angle = [(x1, y1), (x2, y2)], angle
        # return None if we found no eyes fitting the restrictions
        if len(self.eyes_coords) == 0 or self.eye_angle == 31:
            return None
        return True

    def normalize(self):
        """
        Rotates splice of face around midpoint of eyes, updates class values
        Args:
        Returns: True if successful
        """
        self.eye_midpoint = midpointOf(self.eyes_coords)
        rot_mat = cv2.getRotationMatrix2D(self.eye_midpoint, self.eye_angle, 1.0)
        self.norm_image = cv2.warpAffine(
                src=self.splice,
                M=rot_mat,
                dsize=(self.width, self.height),
                flags=cv2.INTER_LINEAR
        )
        return True

    def predictTraitValue(self, imagelist_file='image_list.txt', class_label=1, outfile='output', norm=-1, debug=False):
        """
        Calls external functions to obtain predicted value between 0 and 1 for given image, updates class values
        Args:
            imagelist_file: file that contains the list of image/images to be processed
            class_label: label given images as either positive or negative for SVM classification
            outfile: filename of output for Dense SIFT analysis
        Return: True if successful
        """
        with open(imagelist_file, 'w') as f:
                f.write('tmp.jpg\n')
        cv2.imwrite('tmp.jpg', self.norm_image)

        hog_histogram(imagelist_file, class_label, outfile, norm, debug)

        prob_y, prob_x = svm_read_problem(outfile)
        model_file = models[self.trait]
        model = load_model(model_file)
        self.p_val = predict(prob_y, prob_x, model)[2][0][0]

        return True

    def normalizeScore(self):
        """
        basic normalize score for better graphic representation-
        dominance: [.3, .8] map to [0, 1]
        trustworthiness: [0, .6] map to [0, 1]
        """
        if self.p_val != -1:
            if self.trait == 'Dominance':
                if self.p_val > .8:
                    self.p_val=1
                else:
                    self.p_val = round((self.p_val - .3) * (1/(.8-.3)), 2) if self.p_val >= .3 else 0
            if self.trait == 'Trustworthiness':
                if self.p_val > .7:
                    self.p_val = 1
                else:
                    self.p_val = round((self.p_val - 0)  * (1/(.7-0)), 2) if self.p_val > 0 else 0
        return True

    def predictValue(self):
        if self.findEyes() and self.normalize() and self.predictTraitValue() and self.normalizeScore():
            return True
        return False

