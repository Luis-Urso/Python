
################################################################################################################################################
## SIGN LANGUAGE INTERPRETER (SLI)
## Version 4.0 - 04-JAN-2023
## By Luis A. Urso
## LUCA2.AI (R) 
##
## Version Improvements/Corrections:
##
## 31-Dec_2022:
## - Pearson Correlation implemented for X and Y axis to improve signal changes recognition
## - Bias Inclusip for Moviment Analysis Threshold by Axis (+bias)
##
## 01-JAN-2023:
## - Code improvements
## - Correlation Factor adjustment from 0.97 to 0.972
##
## 04-jan-2023
##
## - Result Screen coding
## - Funciton implementation (code organization)
## - Improve moviments logics + coordinates reset
## - Included the hyperparameters for Hands object creation (hands)
## - Split of Weighted Average parameters by axis
## 
##
## Backlog:
## - Improve the XVZ Mean and Correl Related Parameters Adjustments
## - Implement Z axis resolution (need to define the best conversion factor)
## - Implement the Machine Learning Layer - see references at: https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
## 
################################################################################################################################################

import cv2
import mediapipe as mp 
import time
import numpy as np


###
### Main Fuintion - Activation Logic
###

def main():

    cap = cv2.VideoCapture(0)

    # Get the WebCam Screen Sizes 

    wb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    wb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Creates the Response Screen

    resp_zoom = 1
    
    rsp_img = np.zeros((int(wb_h*resp_zoom),int(wb_w*resp_zoom),3), np.uint8)
    cv2.imshow('Response Screen',rsp_img)

    # Moviment Analysis Threshold Variables - Filter 1 - Average Methof

    th_x = 4
    th_y = 4
    th_z = 4

    # Moviment Analysis Threshold Variables - Filter 2 - Pearson Correl

    th_corr_x = 0.98
    th_corr_y = 0.98
    th_corr_z = 0.98
    
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

    w_mov_avg_x = [1,1,20,20,1000,1,20,20,1000,1,1,1,15,1,1,1,15,1,1,1,15] 
    w_mov_avg_y = [1,1,20,20,1000,1,20,20,1000,1,1,1,15,1,1,1,15,1,1,1,15]
    
    #
    # Recording Buffer - Last "n" Moviments recorded to be analized by the Neural Network
    # In the 1st Order: "n" - is the size of the buffer (e.g. If framerate is 30 frames per second, 
    # use 90 for 3 last 3 seconds)
    # In the 2nd order it is stored the 21 Landmarks, and coordinares X,Y,Z
    #
    
    rec_buffer_size = 90
    
    rec_buffer_frame=[]
    rec_buffer_id=[]
    rec_buffer_cx=[]
    rec_buffer_cy=[]
    rec_buffer_cz=[]
    
    i_buffer=0
    
	# Other Control Variables 

    f_changed=False
    cz=0


    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles


    while True:
        success, cap_img = cap.read()
        cap_imgRBG = cv2.cvtColor(cap_img, cv2.COLOR_BGR2RGB)
        results = hands.process(cap_imgRBG)
        #print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            
            for handLms in results.multi_hand_landmarks:
                
                mpDraw.draw_landmarks(cap_img,handLms,mpHands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())		
        
                for id, lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    h, w, c = cap_img.shape
                    cx, cy = (int(lm.x * w), int(lm.y * h))
                    cz = (lm.z / w)
        
                    #if id==0:
                        #print(id,cz)
        
                    cur_cx[id]=cx
                    cur_cy[id]=cy
                    
                    ## Creates the Record Buffer (memory buffer - last "n-frames" movements)
                    
                    if id==20:                  
                        if i_buffer<=rec_buffer_size:
                            rec_buffer_frame.append(i_buffer)
                            rec_buffer_id.append(id)
                            rec_buffer_cx=cur_cx
                            rec_buffer_cy=cur_cy
                            rec_buffer_cz.append(cz)
                        else:
                            rec_buffer_frame=rec_buffer_frame[1:rec_buffer_size]
                            rec_buffer_id=rec_buffer_id[1:rec_buffer_size]
                            rec_buffer_cx=rec_buffer_cx[1:rec_buffer_size]
                            rec_buffer_cy=rec_buffer_cy[1:rec_buffer_size]
                            rec_buffer_cz=rec_buffer_cz[1:rec_buffer_size]
                            
                            rec_buffer_frame.append(i_buffer)
                            rec_buffer_id.append(id)
                            rec_buffer_cx.append(cx)
                            rec_buffer_cy.append(cy)
                            rec_buffer_cz.append(cz)
                                        

                 
        
                    if id==20:
                        
                        i_buffer+=1
                        
                        mean_prv_cx=np.average(prv_cx,axis=0,weights=w_mov_avg_x)
                        #mean_prv_cx=np.mean(prv_cx)
                        mean_cur_cx=np.average(cur_cx,axis=0,weights=w_mov_avg_x)
                        
                        mean_prv_cy=np.average(prv_cy,axis=0,weights=w_mov_avg_y)
                        #mean_prv_cy=np.mean(prv_cy)
                        mean_cur_cy=np.average(cur_cy,axis=0,weights=w_mov_avg_y)

                        if (mean_cur_cx>(mean_prv_cx+th_x)) or (mean_cur_cx<(mean_prv_cx-th_x)) or (mean_cur_cy>(mean_prv_cy+th_y)) or (mean_cur_cy<(mean_prv_cy-th_y)):
            
                            correl_cx=np.corrcoef(cur_cx,prv_cx)
                            correl_cy=np.corrcoef(cur_cy,prv_cy)
                                                  
                            print("Thresholds (X,Y,Z):",th_x,th_y,th_z)
                            print("X Change Average Vector P->N , DIFF = ", mean_prv_cx,mean_cur_cx,mean_prv_cx-mean_cur_cx )
                            print("Y Change Average Vector P->N , DIFF = ", mean_prv_cy,mean_cur_cy,mean_prv_cy-mean_cur_cy)
                            print("Correl X = ", correl_cx[0,1])
                            print("Correl Y = ",correl_cy[0,1])
                            
                                                   
                            if correl_cx[0,1]<=th_corr_x  or correl_cy[0,1]<=th_corr_x:
                                f_changed = True
                                print("*** Changed Position ***")
                                
                                ## Shows the Hand's Mimic at Response Screen 
                                
                                # build_resp_screen(rsp_img,wb_w,wb_h,cur_cx,cur_cy,resp_zoom)
                              
                                play_buffer(rsp_img,wb_w,wb_h,rec_buffer_cx,rec_buffer_cy,rec_buffer_cz,i_buffer,resp_zoom)
                                
                               
                                i_buffer=0
                                rec_buffer_frame=[]
                                rec_buffer_id=[]
                                rec_buffer_cx=[]
                                rec_buffer_cy=[]
                                rec_buffer_cz=[]

                                
                                
                               
                            prv_cx=cur_cx
                            prv_cy=cur_cy
                            cur_cx=np.zeros(21,dtype=int)
                            cur_cy=np.zeros(21,dtype=int)
    
        
                    #print(id, cx, cy)
                    
                    
        
        # Frame Rate Calculation 

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
    
        if f_changed:
            cv2.putText(cap_img,"*",(10,100),cv2.FONT_HERSHEY_PLAIN,10,(234,242,7),3)
            f_changed=False
    
        cv2.imshow("Capture Screen",cv2.flip(cap_img, 1))
        
        if cv2.waitKey(1) & 0xFF == ord('b'):
            print("***")
            print("--------------BUFFER HERE------------------")
            print("***")
            print(rec_buffer_frame,"\n")
        
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit()
        

##
## Function to Show Static Hand Position - Last Position 
##
        
def build_resp_screen(rsp_img,w_size,h_size,x,y,resp_zoom):
     
    rsp_img = np.zeros((int(h_size*resp_zoom),int(w_size*resp_zoom),3), np.uint8)
    
    circle_size=7
    line_tick=2
    
    # Makes the Hand's Joints
    
    #### PALM JOINTS
    
    cv2.circle(rsp_img,(int(x[0]*resp_zoom),int(y[0]*resp_zoom)),circle_size,(12,12,237),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[5]*resp_zoom),int(y[5]*resp_zoom)),circle_size,(12,12,237),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[9]*resp_zoom),int(y[9]*resp_zoom)),circle_size,(12,12,237),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[13]*resp_zoom),int(y[13]*resp_zoom)),circle_size,(12,12,237),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[17]*resp_zoom),int(y[17]*resp_zoom)),circle_size,(12,12,237),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[1]*resp_zoom),int(y[1]*resp_zoom)),circle_size,(12,12,237),cv2.FILLED)
    
    #### THUMB JOINTS
    
    cv2.circle(rsp_img,(int(x[2]*resp_zoom),int(y[2]*resp_zoom)),circle_size,(161,209,227),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[3]*resp_zoom),int(y[3]*resp_zoom)),circle_size,(161,209,227),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[4]*resp_zoom),int(y[4]*resp_zoom)),circle_size,(161,209,227),cv2.FILLED)
    
    #### INDEX JOINTS
    
    cv2.circle(rsp_img,(int(x[6]*resp_zoom),int(y[6]*resp_zoom)),circle_size,(237,17,193),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[7]*resp_zoom),int(y[7]*resp_zoom)),circle_size,(237,17,193),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[8]*resp_zoom),int(y[8]*resp_zoom)),circle_size,(237,17,193),cv2.FILLED)
    
    #### MIDDLE JOINTS
    
    cv2.circle(rsp_img,(int(x[10]*resp_zoom),int(y[10]*resp_zoom)),circle_size,(17,222,237),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[11]*resp_zoom),int(y[11]*resp_zoom)),circle_size,(17,222,237),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[12]*resp_zoom),int(y[12]*resp_zoom)),circle_size,(17,222,237),cv2.FILLED)
    
    #### RING JOINTS
    
    cv2.circle(rsp_img,(int(x[14]*resp_zoom),int(y[14]*resp_zoom)),circle_size,(2,247,23),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[15]*resp_zoom),int(y[15]*resp_zoom)),circle_size,(2,247,23),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[16]*resp_zoom),int(y[16]*resp_zoom)),circle_size,(2,247,23),cv2.FILLED)
    
    
    ### LITTLE JOINTS
    
    cv2.circle(rsp_img,(int(x[18]*resp_zoom),int(y[18]*resp_zoom)),circle_size,(230,88,32),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[19]*resp_zoom),int(y[19]*resp_zoom)),circle_size,(230,88,32),cv2.FILLED)
    cv2.circle(rsp_img,(int(x[20]*resp_zoom),int(y[20]*resp_zoom)),circle_size,(230,88,32),cv2.FILLED)
    
    # Makes the Hans's Connections 
    
    #### PALM CONNECTIONS
    
    cv2.line(rsp_img,(int(x[0]*resp_zoom),int(y[0]*resp_zoom)),(int(x[5]*resp_zoom),int(y[5]*resp_zoom)),color=(143,148,147),thickness=line_tick)
    cv2.line(rsp_img,(int(x[0]*resp_zoom),int(y[0]*resp_zoom)),(int(x[17]*resp_zoom),int(y[17]*resp_zoom)),color=(143,148,147),thickness=line_tick)
    cv2.line(rsp_img,(int(x[5]*resp_zoom),int(y[5]*resp_zoom)),(int(x[9]*resp_zoom),int(y[9]*resp_zoom)),color=(143,148,147),thickness=line_tick)
    cv2.line(rsp_img,(int(x[9]*resp_zoom),int(y[9]*resp_zoom)),(int(x[13]*resp_zoom),int(y[13]*resp_zoom)),color=(143,148,147),thickness=line_tick)
    cv2.line(rsp_img,(int(x[13]*resp_zoom),int(y[13]*resp_zoom)),(int(x[17]*resp_zoom),int(y[17]*resp_zoom)),color=(143,148,147),thickness=line_tick)
    cv2.line(rsp_img,(int(x[0]*resp_zoom),int(y[0]*resp_zoom)),(int(x[1]*resp_zoom),int(y[1]*resp_zoom)),color=(143,148,147),thickness=line_tick)
    
    #### THUMB CONNECTIONS
    
    cv2.line(rsp_img,(int(x[1]*resp_zoom),int(y[1]*resp_zoom)),(int(x[2]*resp_zoom),int(y[2]*resp_zoom)),color=(161,209,227),thickness=line_tick)
    cv2.line(rsp_img,(int(x[2]*resp_zoom),int(y[2]*resp_zoom)),(int(x[3]*resp_zoom),int(y[3]*resp_zoom)),color=(161,209,227),thickness=line_tick)
    cv2.line(rsp_img,(int(x[3]*resp_zoom),int(y[3]*resp_zoom)),(int(x[4]*resp_zoom),int(y[4]*resp_zoom)),color=(161,209,227),thickness=line_tick)
    
    
    #### INDEX CONNECTIONS
    
    cv2.line(rsp_img,(int(x[5]*resp_zoom),int(y[5]*resp_zoom)),(int(x[6]*resp_zoom),int(y[6]*resp_zoom)),color=(237,17,193),thickness=line_tick)
    cv2.line(rsp_img,(int(x[6]*resp_zoom),int(y[6]*resp_zoom)),(int(x[7]*resp_zoom),int(y[7]*resp_zoom)),color=(237,17,193),thickness=line_tick)
    cv2.line(rsp_img,(int(x[7]*resp_zoom),int(y[7]*resp_zoom)),(int(x[8]*resp_zoom),int(y[8]*resp_zoom)),color=(237,17,193),thickness=line_tick)
    
    
    #### MIDDLE CONNECTIONS
    
    cv2.line(rsp_img,(int(x[9]*resp_zoom),int(y[9]*resp_zoom)),(int(x[10]*resp_zoom),int(y[10]*resp_zoom)),color=(17,222,237),thickness=line_tick)
    cv2.line(rsp_img,(int(x[10]*resp_zoom),int(y[10]*resp_zoom)),(int(x[11]*resp_zoom),int(y[11]*resp_zoom)),color=(17,222,237),thickness=line_tick)
    cv2.line(rsp_img,(int(x[11]*resp_zoom),int(y[11]*resp_zoom)),(int(x[12]*resp_zoom),int(y[12]*resp_zoom)),color=(17,222,237),thickness=line_tick)
	
	#### RING CONNECTIONS
 
    cv2.line(rsp_img,(int(x[13]*resp_zoom),int(y[13]*resp_zoom)),(int(x[14]*resp_zoom),int(y[14]*resp_zoom)),color=(2,247,23),thickness=line_tick)
    cv2.line(rsp_img,(int(x[14]*resp_zoom),int(y[14]*resp_zoom)),(int(x[15]*resp_zoom),int(y[15]*resp_zoom)),color=(2,247,23),thickness=line_tick)
    cv2.line(rsp_img,(int(x[15]*resp_zoom),int(y[15]*resp_zoom)),(int(x[16]*resp_zoom),int(y[16]*resp_zoom)),color=(2,247,23),thickness=line_tick) 
 
	#### LITTLE CONNECTIONS
 
    cv2.line(rsp_img,(int(x[17]*resp_zoom),int(y[17]*resp_zoom)),(int(x[18]*resp_zoom),int(y[18]*resp_zoom)),color=(230,88,32),thickness=line_tick)
    cv2.line(rsp_img,(int(x[18]*resp_zoom),int(y[18]*resp_zoom)),(int(x[19]*resp_zoom),int(y[19]*resp_zoom)),color=(230,88,32),thickness=line_tick)
    cv2.line(rsp_img,(int(x[19]*resp_zoom),int(y[19]*resp_zoom)),(int(x[20]*resp_zoom),int(y[20]*resp_zoom)),color=(230,88,32),thickness=line_tick) 
    
 
    cv2.imshow("Response Screen",cv2.flip(rsp_img, 1))
    
    

def play_buffer(rsp_img,w_size,h_size,rec_buffer_cx,rec_buffer_cy,rec_buffer_cz,rec_buffer_size,resp_zoom):
    
    rsp_img = np.zeros((int(h_size*resp_zoom),int(w_size*resp_zoom),3), np.uint8)
    
    circle_size=7
    line_tick=2
    
    x=np.zeros(21,dtype=np.uint8)
    y=np.zeros(21,dtype=np.uint8)
    z=np.zeros(21,dtype=np.uint8)
  
    
    for frame in range(rec_buffer_size):
        
        for id in range(0,20):
           x[id]=rec_buffer_cx[id]
           y[id]=rec_buffer_cy[id]
           z[id]=rec_buffer_cz[id]
           
    
    
        # Makes the Hand's Joints
        
        #### PALM JOINTS
        
        cv2.circle(rsp_img,(int(x[0]*resp_zoom),int(y[0]*resp_zoom)),circle_size,(12,12,237),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[5]*resp_zoom),int(y[5]*resp_zoom)),circle_size,(12,12,237),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[9]*resp_zoom),int(y[9]*resp_zoom)),circle_size,(12,12,237),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[13]*resp_zoom),int(y[13]*resp_zoom)),circle_size,(12,12,237),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[17]*resp_zoom),int(y[17]*resp_zoom)),circle_size,(12,12,237),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[1]*resp_zoom),int(y[1]*resp_zoom)),circle_size,(12,12,237),cv2.FILLED)
        
        #### THUMB JOINTS
        
        cv2.circle(rsp_img,(int(x[2]*resp_zoom),int(y[2]*resp_zoom)),circle_size,(161,209,227),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[3]*resp_zoom),int(y[3]*resp_zoom)),circle_size,(161,209,227),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[4]*resp_zoom),int(y[4]*resp_zoom)),circle_size,(161,209,227),cv2.FILLED)
        
        #### INDEX JOINTS
        
        cv2.circle(rsp_img,(int(x[6]*resp_zoom),int(y[6]*resp_zoom)),circle_size,(237,17,193),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[7]*resp_zoom),int(y[7]*resp_zoom)),circle_size,(237,17,193),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[8]*resp_zoom),int(y[8]*resp_zoom)),circle_size,(237,17,193),cv2.FILLED)
        
        #### MIDDLE JOINTS
        
        cv2.circle(rsp_img,(int(x[10]*resp_zoom),int(y[10]*resp_zoom)),circle_size,(17,222,237),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[11]*resp_zoom),int(y[11]*resp_zoom)),circle_size,(17,222,237),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[12]*resp_zoom),int(y[12]*resp_zoom)),circle_size,(17,222,237),cv2.FILLED)
        
        #### RING JOINTS
        
        cv2.circle(rsp_img,(int(x[14]*resp_zoom),int(y[14]*resp_zoom)),circle_size,(2,247,23),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[15]*resp_zoom),int(y[15]*resp_zoom)),circle_size,(2,247,23),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[16]*resp_zoom),int(y[16]*resp_zoom)),circle_size,(2,247,23),cv2.FILLED)
        
        
        ### LITTLE JOINTS
        
        cv2.circle(rsp_img,(int(x[18]*resp_zoom),int(y[18]*resp_zoom)),circle_size,(230,88,32),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[19]*resp_zoom),int(y[19]*resp_zoom)),circle_size,(230,88,32),cv2.FILLED)
        cv2.circle(rsp_img,(int(x[20]*resp_zoom),int(y[20]*resp_zoom)),circle_size,(230,88,32),cv2.FILLED)
        
        # Makes the Hans's Connections 
        
        #### PALM CONNECTIONS
        
        cv2.line(rsp_img,(int(x[0]*resp_zoom),int(y[0]*resp_zoom)),(int(x[5]*resp_zoom),int(y[5]*resp_zoom)),color=(143,148,147),thickness=line_tick)
        cv2.line(rsp_img,(int(x[0]*resp_zoom),int(y[0]*resp_zoom)),(int(x[17]*resp_zoom),int(y[17]*resp_zoom)),color=(143,148,147),thickness=line_tick)
        cv2.line(rsp_img,(int(x[5]*resp_zoom),int(y[5]*resp_zoom)),(int(x[9]*resp_zoom),int(y[9]*resp_zoom)),color=(143,148,147),thickness=line_tick)
        cv2.line(rsp_img,(int(x[9]*resp_zoom),int(y[9]*resp_zoom)),(int(x[13]*resp_zoom),int(y[13]*resp_zoom)),color=(143,148,147),thickness=line_tick)
        cv2.line(rsp_img,(int(x[13]*resp_zoom),int(y[13]*resp_zoom)),(int(x[17]*resp_zoom),int(y[17]*resp_zoom)),color=(143,148,147),thickness=line_tick)
        cv2.line(rsp_img,(int(x[0]*resp_zoom),int(y[0]*resp_zoom)),(int(x[1]*resp_zoom),int(y[1]*resp_zoom)),color=(143,148,147),thickness=line_tick)
        
        #### THUMB CONNECTIONS
        
        cv2.line(rsp_img,(int(x[1]*resp_zoom),int(y[1]*resp_zoom)),(int(x[2]*resp_zoom),int(y[2]*resp_zoom)),color=(161,209,227),thickness=line_tick)
        cv2.line(rsp_img,(int(x[2]*resp_zoom),int(y[2]*resp_zoom)),(int(x[3]*resp_zoom),int(y[3]*resp_zoom)),color=(161,209,227),thickness=line_tick)
        cv2.line(rsp_img,(int(x[3]*resp_zoom),int(y[3]*resp_zoom)),(int(x[4]*resp_zoom),int(y[4]*resp_zoom)),color=(161,209,227),thickness=line_tick)
        
        
        #### INDEX CONNECTIONS
        
        cv2.line(rsp_img,(int(x[5]*resp_zoom),int(y[5]*resp_zoom)),(int(x[6]*resp_zoom),int(y[6]*resp_zoom)),color=(237,17,193),thickness=line_tick)
        cv2.line(rsp_img,(int(x[6]*resp_zoom),int(y[6]*resp_zoom)),(int(x[7]*resp_zoom),int(y[7]*resp_zoom)),color=(237,17,193),thickness=line_tick)
        cv2.line(rsp_img,(int(x[7]*resp_zoom),int(y[7]*resp_zoom)),(int(x[8]*resp_zoom),int(y[8]*resp_zoom)),color=(237,17,193),thickness=line_tick)
        
        
        #### MIDDLE CONNECTIONS
        
        cv2.line(rsp_img,(int(x[9]*resp_zoom),int(y[9]*resp_zoom)),(int(x[10]*resp_zoom),int(y[10]*resp_zoom)),color=(17,222,237),thickness=line_tick)
        cv2.line(rsp_img,(int(x[10]*resp_zoom),int(y[10]*resp_zoom)),(int(x[11]*resp_zoom),int(y[11]*resp_zoom)),color=(17,222,237),thickness=line_tick)
        cv2.line(rsp_img,(int(x[11]*resp_zoom),int(y[11]*resp_zoom)),(int(x[12]*resp_zoom),int(y[12]*resp_zoom)),color=(17,222,237),thickness=line_tick)
        
        #### RING CONNECTIONS
    
        cv2.line(rsp_img,(int(x[13]*resp_zoom),int(y[13]*resp_zoom)),(int(x[14]*resp_zoom),int(y[14]*resp_zoom)),color=(2,247,23),thickness=line_tick)
        cv2.line(rsp_img,(int(x[14]*resp_zoom),int(y[14]*resp_zoom)),(int(x[15]*resp_zoom),int(y[15]*resp_zoom)),color=(2,247,23),thickness=line_tick)
        cv2.line(rsp_img,(int(x[15]*resp_zoom),int(y[15]*resp_zoom)),(int(x[16]*resp_zoom),int(y[16]*resp_zoom)),color=(2,247,23),thickness=line_tick) 
    
        #### LITTLE CONNECTIONS
    
        cv2.line(rsp_img,(int(x[17]*resp_zoom),int(y[17]*resp_zoom)),(int(x[18]*resp_zoom),int(y[18]*resp_zoom)),color=(230,88,32),thickness=line_tick)
        cv2.line(rsp_img,(int(x[18]*resp_zoom),int(y[18]*resp_zoom)),(int(x[19]*resp_zoom),int(y[19]*resp_zoom)),color=(230,88,32),thickness=line_tick)
        cv2.line(rsp_img,(int(x[19]*resp_zoom),int(y[19]*resp_zoom)),(int(x[20]*resp_zoom),int(y[20]*resp_zoom)),color=(230,88,32),thickness=line_tick) 
        
    
        cv2.imshow("Response Screen",cv2.flip(rsp_img, 1))   


        
if __name__ == '__main__':
    main()
    
    
    
### THE END ###