import cv2 
import mediapipe as mp 
import time 

cap = cv2.VideoCapture(0) 

mpHands = mp.solutions.hands 
hands = mpHands.Hands() 
mpDraw = mp.solutions.drawing_utils 

pTime = 0 
cTime = 0 

while True: 
    success, img = cap.read() 
    imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    results = hands.process(imgRBG) 
    print(results.multi_hand_landmarks) 
    if results.multi_hand_landmarks: 
        for handLms in results.multi_hand_landmarks: 
            print("Check1",handLms)
            for id, lm in enumerate(handLms.landmark): 
                # print(id,lm) 
                h, w, c = img.shape
                cx, cy = (int(lm.x * w), int(lm.y * h)) 
                cz = (int(lm.z))
                print(id,lm)
                #print(id, cx, cy, cz) 
                if id == 0: 
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED) 
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) 
                    cTime = time.time() 
                    if (cTime-pTime)>0:
                        fps = 1/(cTime-pTime) 
                        pTime = cTime 
                    else:
                        pTime = 0
                    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3) 
                    cv2.imshow("Image",img) 
                    cv2.waitKey(1)