##
## Sign Languague Interptreter (SLI)
## Version 2.0 - 31-DEC-2022
## By Luis A. Urso
##
## Version Improvements/Corrections:
## - Pearson Correlation implemented for X and Y axis to improve signal changes recognition
## - Bias Inclusip for Moviment Analysis Threshold by Axis (+bias)
##
## Backlog:
## - Implement Z axis resolution (need to define the best conversion factor)
## - Implement the Machine Learning Layer - see references at: https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
## 


import cv2
import mediapipe as mp 
import time
import numpy as np
import random


cap = cv2.VideoCapture(0)

# Get the WebCam Screen Sizes 

wb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
wb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the movement factor / drawing rate

pace = 1

# Moviment Analysis Threshold Variables - Filter 1 - Average Methof

th_global = 10
th_x = th_global + 0
th_y = (th_x*(wb_w/wb_h)) + 0
th_z = th_x + 0

# Moviment Analysis Threshold Variables - Filter 2 - Pearson Correl

th_corr_x = 0.97
th_corr_y = 0.97
th_corr_z = 0.97

# Variables to calculate Frame Rate

pTime = 0
cTime = 0

# Toggle - to activate funcitons using keyboard 

activate = 0

# Variables to measure Previous and Current Movements 

cur_id=np.zeros(21,dtype=int)
cur_cx=np.zeros(21,dtype=int)
cur_cy=np.zeros(21,dtype=int)
cur_cz=np.zeros(21,dtype=int)

prv_id=np.zeros(21,dtype=int)
prv_cx=np.zeros(21,dtype=int)
prv_cy=np.zeros(21,dtype=int)
prv_cz=np.zeros(21,dtype=int)

# Weight Movement Vector

w_movement = [1,1,1,1,15,1,1,1,15,1,1,1,15,1,1,1,15,1,1,1,15]

# Other Control Variables 

f_changed=False
cz=0

##########################################################################

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


while True:
	success, img = cap.read()
	imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(imgRBG)
	#print(results.multi_hand_landmarks)

	if results.multi_hand_landmarks:
		
		for handLms in results.multi_hand_landmarks:
			
			mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())		
	
			for id, lm in enumerate(handLms.landmark):
				# print(id,lm)
				h, w, c = img.shape
				cx, cy = (int(lm.x * w), int(lm.y * h))
				cz = (lm.z / w)
    
				#if id==0:
					#print(id,cz)
    
				cur_cx[id]=cx
				cur_cy[id]=cy
    
				if id==20:

					mean_prv_cx=np.average(prv_cx,axis=0,weights=w_movement)
					mean_cur_cx=np.average(cur_cx,axis=0,weights=w_movement)
					mean_prv_cy=np.average(prv_cy,axis=0,weights=w_movement)
					mean_cur_cy=np.average(cur_cy,axis=0,weights=w_movement)

					if (mean_cur_cx>(mean_prv_cx+th_x)) or (mean_cur_cx<(mean_prv_cx-th_x)) or (mean_cur_cy>(mean_prv_cy+th_y)) or (mean_cur_cy<(mean_prv_cy-th_y)):
         
						correl_cx=np.corrcoef(cur_cx,prv_cx+th_x)
						correl_cy=np.corrcoef(cur_cy,prv_cy+th_y)
      
						prv_cx=cur_cx
						prv_cy=cur_cy
      
						cur_cx=np.zeros(21,dtype=int)
						cur_cy=np.zeros(21,dtype=int)
						
      
						print("Thresholds (X,Y,Z):",th_x,th_y,th_z)
						print("X Change Average Vector P->N = ", mean_prv_cx,mean_cur_cx)
						print("Y Change Average Vector P->N= ", mean_prv_cy,mean_cur_cy)
						print("Correl X = ", correl_cx[0,1])
						print("Correl Y = ",correl_cy[0,1])
						
					
						if correl_cx[0,1]<=th_corr_x  or correl_cy[0,1]<=th_corr_x:
							f_changed = True
							print("*** Changed Position ***")
   
    
				#print(id, cx, cy)
				
				
    
    # Frame Rate Calculation 

	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
 
	if f_changed:
		cv2.putText(img,"*",(10,70),cv2.FONT_HERSHEY_PLAIN,10,(255,0,255),3)
		f_changed=False
  
	cv2.imshow("Image",cv2.flip(img, 1))
	
    
   
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break