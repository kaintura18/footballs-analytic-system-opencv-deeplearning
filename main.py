from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
from ball_assigner import Ball_assighner
from camera_movement import CameraMovement
import numpy as np


def main():

    vedio_path = 'input_vedio\\08fd33_4.mp4'
    output_path = 'output_vedio\\output.mp4'

    #read video frames
    vedio_frames = read_video(vedio_path)
   
    model_path = 'models//best.pt'

    #initialize tracker and get tracks
    tracker = Tracker(model_path)
    tracks = tracker.get_frames(vedio_frames, stub_path='stubs\\tracks_stubs.pkl', read_from_stub=True)

    #objects postions
    tracker.add_position(tracks)

    #camera movement
    camera_move_estimate=CameraMovement(vedio_frames[0])
    camera_movement_frame=camera_move_estimate.get_camera_movement(vedio_frames,read_stub=True,stub_path="stubs\\camera_movement.pkl")
    camera_move_estimate.add_adjust_positions_to_tracks(vedio_frames,camera_movement_frame)
    
    #interpolarte ball tracks
    tracks['ball'] = tracker.ball_interpolate(tracks['ball'])

    #initialize team assigner and assign teams to players
    team_assigner = TeamAssigner()
    team_assigner.assign_teams(vedio_frames[0], tracks['player'][0])

    for frame_idx, player_detections in enumerate(tracks['player']):
        for player_id, player_detections in player_detections.items():
            
            bbox = player_detections['bbox']
            team_id = team_assigner.team_assignment(player_id, vedio_frames[frame_idx], bbox)
            
            tracks['player'][frame_idx][player_id]['team_id'] = team_id
            tracks['player'][frame_idx][player_id]['color'] = team_assigner.team_colors[team_id]

        
            
    #initialize ball assigner and assign ball to players

    ball_assigner = Ball_assighner()
    team_ball_control = []
    for frame_num ,player_tracks in enumerate(tracks['player']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']  
        assigned_player= ball_assigner.assign_ball(player_tracks, ball_bbox)
        
        if assigned_player !=-1:
            tracks['player'][frame_num][assigned_player]['has_ball']= True 
            team_ball_control.append(tracks['player'][frame_num][assigned_player]['team_id'])
        else:
            if frame_num > 0:
                team_ball_control.append(team_ball_control[-1]) 
            else:
                team_ball_control.append(1)  # Default team if no ball is detected in first frame

    team_ball_control= np.array(team_ball_control)

    
    #draw annotation on the video frames
    output_frames = tracker.draw_annonation(vedio_frames, tracks ,team_ball_control)

    #draw camera movement
    output_frames=camera_move_estimate.draw_camera_movement(output_frames,camera_movement_frame)

    save_video(output_frames, output_path)

if __name__ == "__main__":
    main()