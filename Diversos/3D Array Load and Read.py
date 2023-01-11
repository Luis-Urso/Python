import numpy as np

buffer = np.zeros([5,3,3],dtype=np.uint16)
cx = np.zeros(3,dtype=np.uint16)

cx[0]=10
cx[1]=20
cx[2]=30

print(buffer.shape)

for i in range(0,5):
    for f in range(0,3):
        for g in range(0,3):
            buffer[i][0][0]=i
            buffer[i][1][1]=i
            buffer[i][2][2]=i
        

print(buffer)

np.roll(buffer,-2)
        
# buffer = buffer[0:4,:,:]

print(buffer.shape)

print(buffer)


