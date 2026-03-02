import sys
sys.path.append('../')
from utils.bbox_utils import get_bbox_centre, bbox_width, get_min_distance


class Ball_assighner:
    def __init__(self):
        self.max_ball_distance = 50

    def assign_ball(self,player_tracks, ball_bboxes):
            ball_posi= get_bbox_centre(ball_bboxes)

            min_distance = 999
            assigned_player_id = -1

            for player_id, player_info in player_tracks.items():
                 
                 player_box= player_info['bbox']
                 distance_right= get_min_distance((player_box[0],player_box[-1]), ball_posi)
                 distance_left= get_min_distance((player_box[2],player_box[-1]), ball_posi)
                 distance= min(distance_right, distance_left)

                 if distance < self.max_ball_distance:
                      if distance < min_distance:
                           min_distance = distance
                           assigned_player_id = player_id
            return assigned_player_id
    

                 

          