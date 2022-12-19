import cv2
  
img1 = cv2.imread('gfg.png')
img2 = cv2.imread('apple.jpeg')
  
img2 = cv2.resize(img2, img1.shape[1::-1])
  
cv2.imshow("img 1",img1)
  
cv2.waitKey(0)
  
cv2.imshow("img 2",img2)
  
cv2.waitKey(0)
  
choice = 1
  
while (choice) :
  
    alpha = -2
  
    dst = cv2.addWeighted(img1, alpha , img2, 1-alpha, 0)
  
    cv2.imwrite('alpha_mask_.png', dst)
  
    img3 = cv2.imread('alpha_mask_.png')
  
    cv2.imshow("alpha blending 1",img3)
  
    cv2.waitKey(0)
  
    choice = int(input("Enter 1 to continue and 0 to exit"))