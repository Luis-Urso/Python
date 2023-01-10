import numpy as np

buffer = np.zeros([10,3,3],dtype=np.uint16)
cx = np.zeros(3,dtype=np.uint16)

cx[0]=10
cx[1]=20
cx[2]=30

print(buffer.shape)

for i in range(0,10):
    for f in range(0,3):
        for g in range(0,3):
            buffer[i][0][0]=cx[0]
            buffer[i][1][1]=cx[1]
            buffer[i][2][2]=cx[2]
        
        
buffer = buffer[0:9,:,:]

print(buffer.shape)