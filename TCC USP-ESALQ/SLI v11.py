
################################################################################################################################################
## SIGN LANGUAGE INTERPRETER (SLI)
## Version 11 - 09-APR-2023
## By Luis A. Urso
##
## Version Improvements/Corrections (last 3 versions):
## 
## 11-JAN-2023:
## - Implementation of Recording Moviments Buffer to be used by Neural Network Training
## - Numpy Warning Suppression activated
##
## 26-JAN-2023:
## - Change of image mirroring (image invetion on CV2)
## - Inclusion of mode selection + get_mode function
##
## 09-APR-2023:
## - Code Review and optimization
##
## Backlog:
## - Implement Z axis for learning and interpreting model (need to define the best conversion factor)
## - Implement historcal movement analysis (Formula: n - (3 x Frame Rate) -> those represent 3 secs)
##
## References: 
## ASL Hand Signs Letters Reference: https://www.youtube.com/watch?v=cGavOVNDj1s
## ASL Hand Signs Numbers Reference: https://www.youtube.com/watch?v=cJ6UFIP-Vt0
## 
################################################################################################################################################

import os
import sys
import cv2
import mediapipe as mp 
import tensorflow as tf
import time
import numpy as np
import csv

import copy
import argparse
import itertools

##
## Function to get this Script Path 
##

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

app_path=get_script_path()

##
## Hand Sign Classifier
## Receives the Normalized and Scaled Landmarks Vector and Predicts the Label
##
        
class predict_label(object):
    def __init__(
        self,
        model_path=app_path+'/model/training_classifier.tflite',
        num_threads=1
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index


##
## Read the Labels File and store them in a vector
## Set the global vector as labels_class
## 

    
with open(app_path+'/model/labels.csv',
			encoding='utf-8-sig') as f_labels:
	labels_class = csv.reader(f_labels)
	labels_class = [
		row[0] for row in labels_class
	]

##
## Main Loop Function 
##

def main():

    cap = cv2.VideoCapture(0)

    # Get the current WebCam Screen Sizes 

    wb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    wb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Creates the Response Screen
    # Variable resp_zoom defines the size factor of the response screen.

    resp_zoom = 1.4 # Size of the Resp. Screen
    
    rsp_img = np.zeros((int(wb_h*resp_zoom),int(wb_w*resp_zoom),3), np.uint8)
    cv2.imshow('Response Screen',rsp_img)
    
    # Function Activation Flags activation
    # Enable the historical movement analysis to improve the learning of the movements. 
    
    analyze_flag=False
    
    # Supress Numpy Warning - normally in Pearson Correlation when dividing by ZERO or NaN
    
    np.seterr(invalid='ignore')

    # Moviment Analysis Threshold Variables - Filter 1 - Average Method

    th_x = 2
    th_y = 2
    th_z = 4 # reserved for future usage

    # Moviment Analysis Threshold Variables - Filter 2 - Pearson Correlation

    th_corr_x = 0.985
    th_corr_y = 0.985
    th_corr_z = 0.98 # reserved for future usage
    
    # Variables to calculate Frame Rate

    pTime = 0
    cTime = 0

    # Toggle related variables - to activate funcitons using keyboard 

    mode="Interpreting"
    training_label=0
    training_class=""

    # Variables to measure Previous and Current Movements 

    cur_cx=np.zeros(21,dtype=int)
    cur_cy=np.zeros(21,dtype=int)
    cur_cz=np.zeros(21,dtype=int) # Assigned but reserved for future usage

    prv_cx=np.zeros(21,dtype=int)
    prv_cy=np.zeros(21,dtype=int)
    prv_cz=np.zeros(21,dtype=int) # Reserved for future usage

    # Weight Movement Vector definitions (for Weighted Average Filter Usage)

    w_mov_avg_x = [1,1,20,20,500,1,20,20,500,1,1,1,15,1,1,1,15,1,1,1,15] 
    w_mov_avg_y = [1,1,20,20,500,1,20,20,500,1,1,1,15,1,1,1,15,1,1,1,15]
    
    # Recording Buffer Matrix and Definitions
    # Matrix Order: buffer_rec = [buffer_Size,21,3] which: Buffer_Size is defined in the bariable below, 21 is the Landmarks, 3 for XYZ
    
    buffer_size=100
    buffer_rec = np.zeros([buffer_size,21,3],dtype=np.uint16)
    buffer_index=0
    
    # Other Control Variables 

    f_changed=False
    f_action=""
    
    cz=0
    
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles


    while True:
        success, cap_img = cap.read()
        cap_img = cv2.flip(cap_img, 1)
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
                    cz = int(lm.z * w) + 105
               
                    cur_cx[id]=cx
                    cur_cy[id]=cy
                    cur_cz[id]=cz
        
                    if id==20:
                        
                        ## Records the Landmark Buffer for further analyzis by
                        ## Neural Network
                        
                        if buffer_index < buffer_size:
                        
                            for lm_index in range(0,21):
                                buffer_rec[buffer_index][lm_index][0]=cur_cx[lm_index]
                                buffer_rec[buffer_index][lm_index][1]=cur_cy[lm_index]
                                buffer_rec[buffer_index][lm_index][2]=cur_cz[lm_index]
                                
                            buffer_index+=1
                            
                        else:
                            buffer_rec=buffer_rec[1:(buffer_size),:,:]
                            buffer_rec=np.vstack((buffer_rec,np.zeros((1,)+buffer_rec.shape[1:], dtype=np.uint16)))

                            for lm_index in range(0,21):
                                buffer_rec[buffer_size-1][lm_index][0]=cur_cx[lm_index]
                                buffer_rec[buffer_size-1][lm_index][1]=cur_cy[lm_index]
                                buffer_rec[buffer_size-1][lm_index][2]=cur_cz[lm_index]
                            
                                            
                        if mode!="Training":
                        
                            ## Weighted Averages - Filter 1

                            mean_prv_cx=np.average(prv_cx,axis=0,weights=w_mov_avg_x)
                            mean_cur_cx=np.average(cur_cx,axis=0,weights=w_mov_avg_x)                      
                            mean_prv_cy=np.average(prv_cy,axis=0,weights=w_mov_avg_y)
                            mean_cur_cy=np.average(cur_cy,axis=0,weights=w_mov_avg_y)

                            if (mean_cur_cx>(mean_prv_cx+th_x)) or (mean_cur_cx<(mean_prv_cx-th_x)) or (mean_cur_cy>(mean_prv_cy+th_y)) or (mean_cur_cy<(mean_prv_cy-th_y)):
                
                                ## Pearson Correlaton - Filter 2 
                
                                correl_cx=np.corrcoef(cur_cx,prv_cx)
                                correl_cy=np.corrcoef(cur_cy,prv_cy)

                                ## Debugging lines - uncomment for debug                    
                                # print("Thresholds (X,Y,Z):",th_x,th_y,th_z)
                                # print("X Change W. Avg Vector Prv.->Cur. , Diff = ", mean_prv_cx,mean_cur_cx,mean_prv_cx-mean_cur_cx )
                                # print("Y Change W. Ave Vector Prv.->Cur. , Diff = ", mean_prv_cy,mean_cur_cy,mean_prv_cy-mean_cur_cy)
                                # print("Correl. X = ", correl_cx[0,1])
                                # print("Correl. Y = ",correl_cy[0,1])
                                                                                
                                if correl_cx[0,1]<=th_corr_x  or correl_cy[0,1]<=th_corr_y:
                                    f_changed = True
                                    
                                    print("*** Changed Position ***")

                                    ##                            
                                    ## Shows the Hand's Mimic at Response Screen with the Symbol Interpretation 
                                    ##

                                    build_resp_screen(rsp_img,wb_w,wb_h,cur_cx,cur_cy,resp_zoom,mode,training_label,training_class)
                                    
                                    ## Analyze the Movements for Neural Network Training or Interpretation
                                    
                                    if analyze_flag:
                                        
                                        analyze_movements(rsp_img,wb_w,wb_h,buffer_rec,buffer_index,resp_zoom)
                                    
                                        buffer_begin=buffer_rec[buffer_index-2:buffer_index-1,:,:]
                                        buffer_rec = np.zeros([buffer_size,21,3],dtype=np.uint16)
                                        buffer_rec[0,:,:]=buffer_begin[0,:,:]
                                        buffer_index=1    
                                
                                prv_cx=cur_cx                            
                                prv_cy=cur_cy
                                cur_cx=np.zeros(21,dtype=int)
                                cur_cy=np.zeros(21,dtype=int)
                            
        
        ## Modes Selection:
        ## 't' - for training, and during the trainng "c" to capture the hand's landmarks.
        ## 'i' - for interpreting
        ## 'q' - to quit
        
        key=cv2.waitKey(10) 
        
        if key & 0xFF == ord('t'):
            mode="Training"
            t_count=0
            
            print("Training Mode Active")
            print("--------------------")
            print(" ")
            print("Provide a new Label ID and Class Name for Training")
            print(" ")
            
            last_label=len(labels_class)-1
            
            print("Last created Label:", last_label)
            print("Last created Class:", labels_class[-1])
            
            while True:
                training_label=int(input("New Label:"))
                if (training_label > last_label):
                    
                    training_class=input("New Class:")
                    
                    # Write the Class in the LABELS file
                    
                    f_labels_write = open(app_path+'\\model\\labels.csv', 'a')
                    f_labels_write.write(training_class+'\n')
                    f_labels_write.close()
                    
                    
                    break
                
                else:
                    
                    print("Head Up: Label already exist. Same CLASS will be attributed automatically")
                    training_class=labels_class[training_label]
                    
                    break
            
            # Refresh Response Screen 
            
            build_resp_screen(rsp_img,wb_w,wb_h,cur_cx,cur_cy,resp_zoom,mode,training_label,training_class)
                 
            
        if mode=="Training" and (key & 0xFF == ord('c')):
            f_action="c"
            t_count+=1
            
            ## Pre-process Landmarks making a normalization of X,Y coordinates.
                                
            lm_normalized = pre_process_landmark(cur_cx,cur_cy)
            
            write_csv(training_label,lm_normalized)
            
            
        if key & 0xFF == ord('i'):
            mode="Interpreting"
            t_count=0
            
        if key & 0xFF == ord('q'):
            mode="Quiting"
            quit()                     
                                              
                    
        
        # Frame Rate Calculation 

        # cTime = time.time()
        # fps = 1/(cTime-pTime)
        # pTime = cTime
    
        if f_changed:
            cv2.putText(cap_img,">*<",(10,50),cv2.FONT_HERSHEY_PLAIN,3,(234,242,7),2)
            f_changed=False
            
        
        if f_action=='c':
            cv2.putText(cap_img,"+++",(wb_w-120,50),cv2.FONT_HERSHEY_PLAIN,3,(73,3,252),2)
            cv2.putText(cap_img,str(t_count),(wb_w-120,80),cv2.FONT_HERSHEY_PLAIN,3,(73,3,252),2)
            f_action=""
            
        cv2.putText(cap_img,'Mode: ' + mode,(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(234,242,7),1)
        cv2.imshow("Capture Screen",cap_img)
                    
##
## Execute the Pre-Processing of the Landmarks + Normalization & Scaling. 
## and Flatten the vectors to a single dimention
##

def pre_process_landmark(lm_x,lm_y):
    
    landmark_list_vector=[]

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    
    for index in range(0,len(lm_x)):
        
        if index == 0:
            base_x = lm_x[0] 
            base_y = lm_y[0]
            
        landmark_list_vector.append(lm_x[index] - base_x)
        landmark_list_vector.append(lm_y[index] - base_y)
    
    # Normalization
    
    max_value = max(list(map(abs, landmark_list_vector)))

    def normalize_(n):
        return n / max_value

    landmark_list_vector = list(map(normalize_,landmark_list_vector))

    return landmark_list_vector

##
## Write CSV File with Training Data
##

def write_csv(label,landmark_list_vector):
    csv_path=get_script_path()+'\\'+'model\\training_data.csv'
    with open(csv_path, 'a', newline="") as file_handler:
        writer = csv.writer(file_handler)
        writer.writerow([label, *landmark_list_vector])    
                

##
## Response Screen building           
##
        
def build_resp_screen(rsp_img,w_size,h_size,x,y,resp_zoom,mode,training_label,training_class):
     
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
    
 
    if mode=="Training":
        cv2.putText(rsp_img,"Label : "+str(training_label),(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(73,3,252),3)
        cv2.putText(rsp_img,"Class : "+training_class,(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(73,3,252),3)
        
        
    if mode=="Interpreting":
        
        start_time=time.time()

        lm_normalized_xy = pre_process_landmark(x,y)
        label = predict_label(lm_normalized_xy)
        label_index=label
        
        if (label_index>0 and label_index<=len(labels_class)):
            predicted_text=labels_class[label_index]
            cv2.putText(rsp_img,predicted_text,(30,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),3)
            print("Predicted: ",predicted_text," - Inference time: ",(time.time()-start_time))
        else:
            cv2.putText(rsp_img,"Not Identified",(30,75),cv2.FONT_HERSHEY_SIMPLEX,3,(73,3,252),3)
            
    cv2.imshow("Response Screen",rsp_img)

##
## Movement Analysis Screen building
## Considers the N past movements defined at Main Loop function setup (see variable buffer_size)
##

def analyze_movements(rsp_img,w_size,h_size,buffer_rec,buffer_index,resp_zoom):
     
    rsp_img = np.zeros((int(h_size*resp_zoom),int(w_size*resp_zoom),3), np.uint8)
    
    circle_size=5
    line_tick=2
    
    x=np.zeros(21,dtype=int)
    y=np.zeros(21,dtype=int)
    z=np.zeros(21,dtype=int)
    
    for frame in range(0,buffer_index):
        for lm_id in range(0,21):
            x[lm_id]=buffer_rec[frame,lm_id,0]
            y[lm_id]=buffer_rec[frame,lm_id,1]
            z[lm_id]=buffer_rec[frame,lm_id,2]
            
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
        
    
        cv2.imshow("Analyer Response",cv2.flip(rsp_img, 1))

##
## Main Function Call (execution starts here!:)
##
        
if __name__ == '__main__':
    
    
    ## Create the Prediction Label Object 
    
    predict_label=predict_label()
    
    
    ## Call the Main Loop
    
    main()
       
### THE END ###