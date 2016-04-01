import cv2
import numpy as np

traits = [
    'Trustworthiness',
    'Dominance'
]

class Graphics:
    """ draws informative graphics above and below picture"""

    def __init__(self):
        self.trait = 0
        self.coordList = []
        self.maxPVal = 0
        self.totalPVal = 0
        self.updateSpeed = 9
        self.totalFrames = 0
        self.background = cv2.imread('background.jpg')
        self.backBotHeight = 0
        self.backTopHeight = 0
        self.frameWidth = 0
        self.frameHeight = 0
        self.histogram = {0.0:0, .1:0, .2:0, .3:0, .4:0, .5:0, .6:0, .7:0, .8:0, .9:0, 1:0}

    def updateCoordList(self, new_coord):
        """maintain list of previous scores and their graph attributes (position, color) """
        for a in range(len(self.coordList)):
            self.coordList[a] = list(self.coordList[a])
            self.coordList[a][0] -= self.updateSpeed
            self.coordList[a] = tuple(self.coordList[a])
        if new_coord != -1:
            self.coordList = [new_coord] + self.coordList
        new_list = []
        for a in range(len(self.coordList)):
            if self.coordList[a][0] > 30:
                new_list.append(self.coordList[a])
        del self.coordList
        self.coordList = new_list

    @staticmethod
    def getColor(score):
        """ given a score, what should it's color be on the graph? """
        if score == -1:
            return (127, 127, 0)
        green_level = int(score * 255)
        blue_level = 254 - green_level
        return blue_level, green_level, 0

    def getCoord(self, score):
        """given a score, where should it be placed on the graph? """
        if score < 0:
            score = 0
        y_var = int(self.frameHeight - int(score*self.backBotHeight))
        x_var = self.frameWidth - 220
        color = self.getColor(score)
        return x_var, y_var, color

    def updateValues(self, p_val):
        self.maxPVal = max(self.maxPVal, p_val)
        self.totalPVal += p_val
        self.totalFrames += 1

    def resetHist(self):
        for key in self.histogram.iterkeys():
            self.histogram[key]=0

    def addBack(self, frame):
        # add graphics' backgrounds at top and bottom of image
        self.backBotHeight = int(frame.shape[0]/5)
        self.backTopHeight = self.backBotHeight
        background = cv2.resize(self.background, (frame.shape[1], self.backBotHeight))
        background_top = cv2.resize(self.background, (frame.shape[1], self.backTopHeight))
        frame = np.concatenate((background_top, frame), axis=0)
        frame = np.concatenate((frame, background), axis=0)
        return frame

    def drawFaces(self, frame, faceList, trait):
        titleCoords = (0, int(float(self.backTopHeight) * (2.0 / 3.0)))
        titleSize = round(float(self.backTopHeight) / 50.0, 2)
        cv2.putText(frame, traits[trait], titleCoords, cv2.FONT_HERSHEY_SIMPLEX, titleSize, (150, 150, 150), 2)

        for face in faceList:
            #cv2.rectangle(frame, (face.x, face.y+self.backTopHeight), (face.x + face.w, face.y+self.backTopHeight + face.h), (255, 255, 255), 1)
            scoreSize = round(float(face.w / 70.0), 2)
            cv2.rectangle(
                img=frame,
                pt1=(face.x + face.w, face.y + self.backTopHeight + int((1-face.p_val) * face.h)),
                pt2=(face.x + face.w + int(face.w / 2.5), face.y + face.h + self.backTopHeight),
                color=Graphics.getColor(face.p_val),
                thickness=-1
            )
            cv2.rectangle(
                img=frame,
                pt1=(face.x+face.w, face.y+self.backTopHeight),
                pt2=(face.x+face.w + int(face.w / 2.5), face.y + face.h + self.backTopHeight),
                color=(255,255,255),
                thickness=1
            )
            cv2.putText(
                img=frame,
                text=str(round(face.p_val, 1))[1:],
                org=(face.x+face.w, face.y+self.backTopHeight+face.h),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=scoreSize,
                color=(0,0,0),
                thickness=2
            )

        return frame

    def drawTitle(self, frame, trait):
        titleCoords = (0, int(float(self.backTopHeight) * (2.0 / 3.0)))
        fontSize = round(float(self.backTopHeight) / 50.0, 2)
        cv2.putText(frame, trait, titleCoords, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (150, 150, 150), 2)
        return frame

    def drawTimeline(self, frame, score):
        self.frameWidth, self.frameHeight = frame.shape[1], frame.shape[0]
        x1, x2, = 0, self.frameWidth - 220
        color25, color75 = (255, 0, 0), (0, 255, 0)
        color50 = list(
            [int((col25 + col75) / 2) for col25, col75 in zip(color25, color75)])
        lineThick, lineType, dotSize, connectThick = 1, 1, 1, 16

        # graph lines of timeline
        line_heights = (int(.25 * self.backBotHeight), int(.50 * self.backBotHeight),
                        int(.75 * self.backBotHeight))
        line_colors = (color25, color50, color75)
        for dist, col in zip(line_heights, line_colors):
            cv2.line(frame, (x1, self.frameHeight - dist), (x2, self.frameHeight - dist), col,
                     lineThick, lineType)

        # update timeline coordinates with new score
        if score != -1:
            new_coord = self.getCoord(score)
            self.updateCoordList(new_coord)
        else:
            self.updateCoordList(-1)

        # draw points on timeline
        if len(self.coordList) > 1:
            for a in range(1, len(self.coordList)):
                cv2.circle(frame, (self.coordList[a][0], self.coordList[a][1]), dotSize, self.coordList[a][2], -1)
                cv2.line(img=frame,
                         pt1=(self.coordList[a - 1][0], self.coordList[a - 1][1]),
                         pt2=(self.coordList[a][0], self.coordList[a][1]),
                         color=self.coordList[a - 1][2],
                         thickness=connectThick)
        else:
            cv2.circle(frame, (
            self.frameWidth - 220, self.getCoord(score)[1]),
                       dotSize, self.getColor(score), -1)

        return frame

    def drawHist(self, frame, score):
        if score != -1:
            self.histogram[round(score, 1)] += 1
        total_vals = sum(self.histogram.itervalues())
        max_value = max(self.histogram.itervalues())
        scale_factor = 1.0 / (float(max_value) / float(total_vals)) if total_vals != 0 else 1
        perc_histogram = {}
        x_val = self.frameWidth - 220
        if total_vals != 0:
            for key, val in self.histogram.iteritems():
                perc_histogram[key] = round((scale_factor * (self.histogram[key] / float(total_vals))), 2)
            for key in sorted(self.histogram.iterkeys()):
                y_val = int(self.frameHeight - perc_histogram[key] * self.backBotHeight)
                x2_val = x_val + 20
                cv2.rectangle(frame, (x_val, y_val), (x2_val, self.frameHeight), (155, 20, 0), -1)
                cv2.putText(frame, str(key), (x_val, self.frameHeight - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .3,
                            (0, 255, 0))
                x_val = x2_val

        return frame

    def draw(self, frame, faceObj, trait, multi):
        self.frameHeight, self.frameWidth = frame.shape[1], frame.shape[0]

        traitStr = traits[trait]

        # add background
        frame = self.addBack(frame)

        # draw title
        frame = self.drawTitle(frame, traitStr)

        if multi:
            return self.drawFaces(frame, faceObj, trait)

        if faceObj is not None:
            self.updateValues(faceObj.p_val)

        # draw timeline
        score = faceObj.p_val if faceObj is not None else -1
        frame = self.drawTimeline(frame, score)

        # draw histogram
        if faceObj is not None:
            frame = self.drawHist(frame, faceObj.p_val)
        else:
            frame = self.drawHist(frame, -1)

        return frame



    def displayFrame(self, frame, scale, loc):
        frame = cv2.resize(
            frame,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR
        )
        cv2.imshow('Demo', frame)
        cv2.moveWindow('Demo', *loc)


