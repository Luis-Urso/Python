########################################
# Hand Tracking with Mouse Move
# See this for Autogui References: https://datatofish.com/control-mouse-python/

import cv2
import mediapipe as mp 
import time
import pyautogui as gui

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

activate = 0
cx = 0
cy = 0

while True:
	success, img = cap.read()
	imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(imgRBG)
	#print(results.multi_hand_landmarks)

	if results.multi_hand_landmarks:
		for handLms in results.multi_hand_landmarks:
			for id, lm in enumerate(handLms.landmark):
				# print(id,lm)
				h, w, c = img.shape
				cx, cy = (int(lm.x * w), int(lm.y * h))
				# print(id, cx, cy)
				if id == 8:
					cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
					gui.moveTo(cx,cy)
					print(cx)
					
     
				if id == 12 and activate==1:
					cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
     			
				# if cv2.waitKey(1) & 0xFF == ord('c') and id==8:
				# cv2.circle(img,(cx,cy),15,(255,0,0),cv2.FILLED)
			
			mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
			
	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime

	
	cv2.putText(img,str(cx),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

	cv2.imshow("Image",cv2.flip(img, 1))
    
	if cv2.waitKey(1) & 0xFF == ord('t'):
		if activate==1:
			activate=0
		else:
			activate=1
    
    
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break