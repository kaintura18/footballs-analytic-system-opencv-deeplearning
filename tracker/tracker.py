import sys
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import os
import pickle
import cv2
sys.path.append('../')
from utils import get_bbox_centre, bbox_width,get_foot_position
    

class Tracker:
    def __init__ (self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def add_position(self,tracks):
      for object,object_tracks in tracks.items():
          for frame_num,tracks in enumerate(object_tracks):
              for track_id,track_info in tracks.items():
                  bbox=track_info["bbox"]
                  if object== "ball":
                      postition=get_bbox_centre(bbox)
                  else:
                      postition=get_foot_position(bbox)
                  tracks[object][frame_num][track_id]["position"]==postition
      
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            detection_batch = self.model.predict(batch,conf=0.1)  
            detections.extend(detection_batch)
        return detections             
    
    def get_frames(self, frames, stub_path=None, read_from_stub=False):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            # Read tracks from the stub file
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "player": [],
            "ball": [],
            "referee": []
        }

        for frame_num, detection in enumerate(detections):
            class_name=detection.names
            class_name_inv = {v: k for k, v in class_name.items()}
                  
            #to superveision forma
            detection_supervision= sv.Detections.from_ultralytics(detection)

            #track goalkeeper to player
            for object_ind,class_id in enumerate(detection_supervision.class_id):
                if class_name[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = class_name_inv['player']
                  
            detect_and_tracked = self.tracker.update_with_detections(detection_supervision)

            tracks["player"].append({})
            tracks["ball"].append({})
            tracks["referee"].append({})


            for track in detect_and_tracked:
                bbox = track[0].tolist()
                track_id = track[4]
                class_id = track[3]

                if class_id == class_name_inv['player']:
                    tracks["player"][frame_num][track_id] = {'bbox': bbox}

                #
                if class_id == class_name_inv['referee']:
                    tracks["referee"][frame_num][track_id] = {'bbox': bbox}

            for track in detection_supervision:
                bbox = track[0].tolist()
                class_id = track[3]

                if class_id == class_name_inv['ball']:
                    tracks["ball"][frame_num][1] = {'bbox': bbox}

        if stub_path is not None:
            # Save tracks to a stub file
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            
        return tracks
    

    def ball_interpolate(self, ball_posittions):

        ball_posittion=[x.get(1,{}).get('bbox',[]) for x in ball_posittions]
        ball_posittion_df = pd.DataFrame(ball_posittion, columns=['x1', 'y1', 'x2', 'y2'])

        ball_posittion_df.interpolate(method='linear', inplace=True)  #interpolate missing values
        ball_posittion_df.bfill() #backward fill for any remaining missing values

        ball_posittion = [{1: {'bbox': x}} for x in ball_posittion_df.values.tolist()]
         
        return ball_posittion
    
    
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2=int(bbox[3])
        x_center, _ = get_bbox_centre(bbox)
        width = bbox_width(bbox)
        

        cv2.ellipse(
        
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        #rectangle parameters with track id
        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
                )   
            

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_bbox_centre(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame


    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame


    def draw_annonation(self,vedio_frames,tracks,team_ball_control):
        output_frames = []
        for frame_num, frame in enumerate(vedio_frames):
            frame=frame.copy()

            player_dict=tracks["player"][frame_num]
            refree_dict=tracks["referee"][frame_num]
            ball_dict=tracks["ball"][frame_num]

            for track_id, player in player_dict.items():
            
                color = player['color'] if 'color' in player else (0, 255, 0)
                frame=self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    frame=self.draw_traingle(frame, player['bbox'], (0, 255, 0))
                    
            for track_id, refree in refree_dict.items():
                frame=self.draw_ellipse(frame, refree['bbox'], (255, 255, 0), track_id)
            
            for track_id, ball in ball_dict.items():
                frame=self.draw_traingle(frame, ball['bbox'], (0, 0, 255))

            
            frame = self.draw_team_ball_control(frame, frame_num,team_ball_control)

                
            output_frames.append(frame)
        return output_frames
              
            