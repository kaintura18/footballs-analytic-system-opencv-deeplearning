from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self,):
        self.team_colors ={}
        self.player_id={}


    def get_color_clusters(self,image_crop):
        # Reshape the image to be a list of pixels
        pixels_2d = image_crop.reshape(-1, 3)

        # Use KMeans to find the dominant colors
        kmeans = KMeans(n_clusters=2, init='k-means++',n_init=10,random_state=0)

        kmeans.fit(pixels_2d)

        return kmeans
  
        
    def get_player_color(self,frame,bbox):
        
        image_crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half = image_crop[:image_crop.shape[0]//2, :]

        #get color clusters
        kmeans=self.get_color_clusters(top_half)
        labels = kmeans.labels_

        #reshape imgae crop to be a list of pixels
        pixels = labels.reshape(top_half.shape[0], top_half.shape[1])
        corners=[pixels[0,0],pixels[0,-1],pixels[-1,0],pixels[-1,-1]]
                 
        non_player_pixels = max(set(corners), key=corners.count)
        player_pixels = 1 - non_player_pixels

        player_color = kmeans.cluster_centers_[player_pixels]

        return player_color


    def assign_teams(self, frame,players_detections):
        player_colors = []
        for _,player_detections in players_detections.items():
            bbox=player_detections['bbox']
            player_color=self.get_player_color(frame,bbox)
            player_colors.append(player_color)  

        kmeans=KMeans(n_clusters=2, init='k-means++',n_init=10,random_state=0).fit(player_colors)

        self.kmeans=kmeans
        self.team_colors[1]=kmeans.cluster_centers_[0]
        self.team_colors[2]=kmeans.cluster_centers_[1]
        
    def team_assignment(self,player_id,frame,bbox):
        if player_id in self.player_id:
            return self.player_id[player_id]
        else:
            player_color=self.get_player_color(frame,bbox)
            team_id=self.kmeans.predict(player_color.reshape(1, -1))[0]
            team_id+=1

            if player_id==102:#hard coded for player /goalie102 as his color is similar to team 1
                team_id=1
            self.player_id[player_id]=team_id

            return team_id