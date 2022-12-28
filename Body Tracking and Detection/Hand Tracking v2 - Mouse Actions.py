#################################################################################################
# Drawing with MediaPipe																		
# by - Luis A. Urso																				
# Version 2.0
#
# Dev. Backlog:
# - Identify Right or Left Hand - see this: https://www.geeksforgeeks.org/right-and-left-hand-detection-using-python/
#
#################################################################################################


import cv2
import mediapipe as mp 
import time
import numpy as np
import random
import pyautogui as pg


# Set the allowance of Mouse going to not allowed area + PYAUTOGUI fails disable. 

pg.FAILSAFE = False

cap = cv2.VideoCapture(0)

# Get the WebCam Resolution to use as conversion factor.

wb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
wb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Set the Mouse Screen Resolution and adjust the rates to make further conversion

res_w = 1920
res_h = 1080

rate_w = (wb_w/res_w)
rate_h = (wb_h/res_h)

# Set Mouse Move Speed

mspeed = 1.6

# Set variables to calculate the FPS

pTime = 0
cTime = 0

mpHands = mp.solutions.hands

hands = mpHands.Hands(model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils


while True:
	success, img = cap.read()
	imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(imgRBG)

	#print(results.multi_hand_landmarks)
	print(results.multi_handedness)

	if results.multi_hand_landmarks:
		for handLms in results.multi_hand_landmarks:
			for id, lm in enumerate(handLms.landmark):
				# print(id,lm)
				h, w, c = img.shape
				cx, cy = (int(lm.x * w), int(lm.y * h))
				print(id, cx, cy)
    
				if id == 8:
					cv2.circle(img, (cx,cy), 15, (130,50,205), cv2.FILLED)
					pg.moveTo(int(((wb_w-cx)/rate_w)*mspeed),int((cy/rate_h)*mspeed))
    
		
			mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
			

	# Calculate the FrameRate (fps)

	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
   
	cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
	cv2.imshow("Hand Track",cv2.flip(img, 1))
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	if cv2.waitKey(1) & 0xFF == ord('c'):
		pg.moveTo(res_w//2,res_h//2)	