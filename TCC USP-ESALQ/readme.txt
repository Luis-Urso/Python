#################################################################################################################
# SLI v11 - Signal Language Interpreter                                                                         #
# by Luis A. Urso                                                                                               #
# Updated by: 15-APR-2023                                                                                       #
#################################################################################################################

Overview: This solution leverages MediaPipe Hands pipeline to het the 21 hands landmarks those will be used 
          as features to be introduced into a Recurring Neural Network (RNN).
          It was developed in the context of creating a paper for USP-ESALQ MBA course and to be avalable for
          others who are exploring solution to improve people life's. 
          

Interpreting and Acquiring Features: This process is performed by the SLI v11.py application. To start acquiring
features, type "t" and during the process type "c" to collect the hands sign samples. Once finished, type "i" to
return to interpreting mode. 

Training: Once the features are acquired, it is required to run the SLI Trainer v2.ipynb (Python Jupyter) notebook
to train the neural network that will be stored into the model folder, with the name of training_classifier.tflite.
          
