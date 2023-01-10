############################################################
# How to make an Array Buffer with Python Basic            #
# by Luis A. Urso                                          #
############################################################

a=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

size=len(a)

for i in range(1000):

	a=a[1:size]
	a.append(20+i)

	print(a)
	print(len(a))

