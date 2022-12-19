# Based on: 
# https://www.geeksforgeeks.org/blending-of-two-videos-using-python/

# importing necessary packages
import numpy as np
import cv2

# assigning path of foreground video
#path_1 = r"C://Users//Lenovo//Desktop//Python Workshop//z.mp4"
fg = cv2.VideoCapture(0)

# assigning path of background video
path_2 = r"C://Users//Lenovo//Desktop//Python Workshop//v.mp4"
bg = cv2.imread('gfg.png')
h, w = 1080, 1920

while True:

	# Reading the two input videos
	# we have taken "ret" here because the duration
	# of bg video is greater than fg video,
	ret, foreground = fg.read()
	
	# if in your case the situation is opposite
	# then take the "ret" for bg video
	background = bg
	
	# if foreground array is not empty which
	# means actual video is still going on
	if ret:
	
		# creating the alpha mask
		alpha = np.zeros_like(foreground)
		gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
		alpha[:, :, 0] = gray
		alpha[:, :, 1] = gray
		alpha[:, :, 2] = gray

		# converting uint8 to float type
		foreground = foreground.astype(float)
		background = background.astype(float)

		# normalizing the alpha mask inorder
		# to keep intensity between 0 and 1
		alpha = alpha.astype(float)/255

		# multiplying the foreground
		# with alpha matte
		foreground = cv2.multiply(alpha,
								foreground)

		# multiplying the background
		# with (1 - alpha)
		background = cv2.multiply(1.0 - alpha,
								background)

		# adding the masked foreground
		# and background together
		outImage = cv2.add(foreground,
						background)

		# resizing the masked output
		ims = cv2.resize(outImage, (980, 540))

		# showing the masked output video
		cv2.imshow('Blended', ims/255)

		# if the user presses 'q' then the
		# program breaks from while loop
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	# if the actual video is over then there's
	# nothing in the foreground array thus
	# breaking from the while loop
	else:
		break
		
print('Video Blending is done perfectly')

