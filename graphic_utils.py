import cv2
import numpy as np


class Graphics:
    """ draws informative graphics above and below picture"""

    def __init__(self):
        self.coord_list = []
        self.max_p_val = 0
        self.total_p_vals = 0
        self.update_speed = 7
        self.total_frames = 0
        self.background = cv2.imread('background.jpg')
        self.background_bot_height = 0
        self.background_top_height = 100

    def update_coord_list(self, new_coord):
        """maintain list of previous scores and their graph attributes (position, color) """
        for a in range(len(self.coord_list)):
            self.coord_list[a] = list(self.coord_list[a])
            self.coord_list[a][0] -= self.update_speed
            self.coord_list[a] = tuple(self.coord_list[a])
        if new_coord != -1:
            self.coord_list = [new_coord] + self.coord_list
        new_list = []
        for a in range(len(self.coord_list)):
            if self.coord_list[a][0] > 150:
                new_list.append(self.coord_list[a])
        del self.coord_list
        self.coord_list = new_list

    @staticmethod
    def get_color(score):
        """ given a score, what should it's color be on the graph? """
        if score == -1:
            return (127, 127, 0)
        green_level = int(score * 255)
        blue_level = 254 - green_level
        return blue_level, green_level, 0

    def get_coord(self, score, frame_height, frame_width, graphic_height):
        """given a score, where should it be placed on the graph? """
        if score < 0:
            score = 0
        y_var = int(frame_height - int(score*graphic_height))
        x_var = frame_width - 150
        color = self.get_color(score)
        return x_var, y_var, color

    def update_values(self, p_val):
        self.max_p_val = max(self.max_p_val, p_val)
        self.total_p_vals += p_val
        self.total_frames += 1

    def add_background(self, frame):
        # add graphics' backgrounds at top and bottom of image
        self.background_bot_height = int(frame.shape[0]/5)
        background = cv2.resize(self.background, (frame.shape[1], self.background_bot_height))
        background_top = cv2.resize(self.background, (frame.shape[1], self.background_top_height))
        frame = np.concatenate((background_top, frame), axis=0)
        frame = np.concatenate((frame, background), axis=0)
        return frame

    @staticmethod
    def draw_graphis_multi_faces(frame, trait, score, face_coords, debug):
        x, y, w, h = face_coords
        if debug:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 1)
        cv2.putText(
            img=frame,
            text=str(round(score,2)),
            org=(x,y-20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=Graphics.get_color(round(score,2)),
            thickness=3
        )
        cv2.putText(
            img=frame,
            text=trait,
            org=(x,y-80),
            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
            fontScale=1,
            color=Graphics.get_color(round(score,2)),
            thickness=1
        )

    def draw_graphics_single_face(self, media_type, frame, score, trait):
        """
        main drawing function, draws graphics on frame to be displayed
        args:   media_type: 1 for webcam capture, 2 for prerecorded video, 3 for still image
                frame: frame to be processed
                score: trait score between 0 and 1 to two decimal places
                trait: Name of trait, i.e. 'Trustworthiness'
                multi_faces: True for finding multiple faces
        returns: new frame
        """
        self.update_values(score)
        frame = self.add_background(frame)

        # general variables:
        # shape of entire window:
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        # shape of bottom or top displays:
        # start and end x coordinate of timeline on bottom graphic:
        timeline_x1, timeline_x2, = 150, frame_width - 150
        # .25, .75, .5 marker lines on timeline:
        timeline_25_color, timeline_75_color = (255, 0, 0), (0, 255, 0)
        timeline_50_color = list([int((col25+col75)/2) for col25, col75 in zip(timeline_25_color, timeline_75_color)])
        timeline_line_thickness, timeline_line_type, timeline_dot_size, timeline_connector_thickness = 1, 1, 5, 7

        # draw graphics:
        # set appropriate title text for media type
        if media_type == 1:  # for webcam capture
            title_text = [
                'Current Trait:  ' + trait,
                'Indicators:  Score out of 1,  Graph over time,  Max and current score',
                'Keys:  n =next trait,  r =reset,  d =debug,  1...9 =graph speed'
            ]
            title_text_coords = [(0, 30), (0, 60), (0, 80)]
        elif media_type == 2:  # for video
            title_text = [
                'Current Trait:  ' + trait,
                'Indicators:  Score out of 1,  Graph over time,  Max and current score'
            ]
            title_text_coords = [(0, 40), (0, 80)]
        else:  # for still images
            title_text = [trait]
            title_text_coords = [(0, 60)]

        #put title text
        for loc, text in zip(title_text_coords, title_text):
            cv2.putText(frame, text, loc, cv2.FONT_HERSHEY_SIMPLEX, .5, (150, 150, 150), 1)

        # put score in bottom left hand corner
        score_str = '0.-' if (score == -1) else str(score)
        cv2.putText(frame, score_str, (0, frame_height-30), cv2.FONT_HERSHEY_SIMPLEX, 2, self.get_color(score), 3)

        if media_type != 3:  #for videos and webcam capture
            # graph lines of timeline
            line_heights = (int(.25*self.background_bot_height), int(.50*self.background_bot_height),int(.75*self.background_bot_height))
            line_colors = (timeline_25_color, timeline_50_color, timeline_75_color)
            for dist, col in zip(line_heights, line_colors ):
                cv2.line(frame, (timeline_x1, frame_height-dist), (timeline_x2, frame_height-dist), col, timeline_line_thickness, timeline_line_type)

            # update timeline coordinates with new score
            if score != -1:
                new_coord = self.get_coord(score, frame_height, frame_width, self.background_bot_height)
                self.update_coord_list(new_coord)
            else:
                self.update_coord_list(-1)

            # draw points on timeline
            if len(self.coord_list) > 1:
                for a in range(1, len(self.coord_list)):
                    cv2.circle(frame, (self.coord_list[a][0], self.coord_list[a][1]), timeline_dot_size, self.coord_list[a][2], -1)
                    cv2.line(frame, (self.coord_list[a-1][0], self.coord_list[a-1][1]), (self.coord_list[a][0], self.coord_list[a][1]), self.coord_list[a-1][2], timeline_connector_thickness)
            else:
                cv2.circle(frame, (frame_width-150, self.get_coord(score, frame_height, frame_width, self.background_bot_height)[1]), timeline_dot_size, self.get_color(score), -1)

            # draw max box and current score box
            cv2.rectangle(frame, (frame_width-100, frame_height-int(self.background_bot_height*self.max_p_val)), (frame_width-50, frame_height), (0,255,0), -1)
            if score != -1:
                cv2.rectangle(frame, (frame_width-95, frame_height-int(score*self.background_bot_height)), (frame_width-55, frame_height), self.get_color(score), -1)

        return frame


