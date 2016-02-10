#! /usr/bin/python2.7
'''
MSER detector demo
==================

Usage:
------
	mser.py img horizon_line focal_length
  
Keys:
-----
	ESC   - exit

'''

import numpy as np
import cv2
import sys

#depth function and cam calibration and collect images
#kuda functions in opencv


#def refl_thresh(focal_length, cam_height, distance_away, height_object):
#	theta = 2*np.arctan(.5*(cam_height/focal_length))
#	height_image = np.arctan(theta)*distance_away -height_object
#	return height_image

def depth_thresh(cam_height, focal_length, image_height): #returns max depth in pixels
	max_distance = 2*focal_length*cam_height
	min_depth = horizon_line + (max_distance**.5)
	max_depth = horizon_line +(max_distance/5)
	if max_depth >	image_height:
		max_depth = image_height
	return min_depth, max_depth

def normed(img,con,sz):
	"""Resizes patch to tuple sz"""
	patch=getPatch(img,con)
	n=cv2.resize(patch, sz)
	return n

def getPatch(img,con):
	"""Returns a patch from the contour"""
	mask=np.zeros(img.shape,dtype=img.dtype)
	cv2.fillConvexPoly(mask,con,(255,255,255))
	mask&=img
	x,y,w,h=cv2.boundingRect(con)
	return mask[y:y+h,x:x+w]

if __name__ == '__main__':
	try: 
		img_name = sys.argv[1]
		horizon_line = int(sys.argv[2])
		focal_length = int(sys.argv[3])
	except: 
		img_name = 0
	img = cv2.imread(img_name)
	mser = cv2.MSER(_min_area=15*15,_max_area=64*48)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	vis = img.copy()
	h,w,ch=img.shape
	regions = mser.detect(gray, None)
	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	normedPatches=[normed(img,p,(100,100)) for p in hulls]
	source = []
	reflect = []
	for i in range(len(hulls)):
		heightavg = (hulls[i].max(axis=0) + hulls[i].max(axis=0))/2
		heightmax = hulls[i].max(axis=0)
		heightmin = hulls[i].min(axis=0)
		error = .1*horizon_line
		if heightmax[0,1] < horizon_line - error:
			source.append(hulls[i])
		elif heightmin[0,1] > horizon_line+error:
			reflect.append(hulls[i])
	for i in range(len(source)): #hulls above horizon line
		dim1 = source[i].max(axis=0) - source[i].min(axis=0)
		display = vis.copy()
		mSrc=cv2.moments(source[i])
		muSrc=np.array([ mSrc['m10']/mSrc['m00'], mSrc['m01']/mSrc['m00'] ])
		src=normedPatches[i]
		srcFlipped=cv2.flip(src,0)
		maxScore=0
		maxIdx=-1
		for j in range(len(reflect)): #hulls below horizon line
			if i!=j:
				dim2 = reflect[j].max(axis=0) - reflect[j].min(axis=0)
				mRef=cv2.moments(reflect[j])
				muRef=np.array([ mRef['m10']/mRef['m00'],
								mRef['m01']/mRef['m00'] ])
				d=muRef-muSrc
				if abs(d[0])<6 and abs(d[1])>8: #x is d[0], y is d[1]
					width_upperbound = dim2*1.1
					width_lowerbound = dim2*.9
					if dim1[0,0] > width_lowerbound[0,0] and dim1[0,0] < width_upperbound[0,0]:
						height_lowerbound = (2*horizon_line)- muSrc[1]
						if muRef[1] > height_lowerbound:
							min_depth, max_depth = depth_thresh(2, focal_length, h)
							#print(muRef[1], min_depth, max_depth)
							if muRef[1] < max_depth and muRef[1] > min_depth:
								corrRes=cv2.matchTemplate(srcFlipped,normedPatches[j],cv2.TM_CCOEFF) #TM_CCOEFF are effective methods
								score=corrRes.max()
								if score>maxScore:
									maxIdx=j
									maxScore=score
		if maxIdx!=-1 and maxScore>0.9:
			color = [np.random.uniform(0,255), np.random.uniform(0,255), np.random.uniform(0,255)]			
			cv2.drawContours(display,source,i,(255,255,255),2)
			cv2.drawContours(display,reflect,maxIdx,(0,255,255),2)
			#cv2.namedWindow("reflection1.png", cv2.WINDOW_NORMAL)
			cv2.imshow("reflection1.png", display)
			cv2.waitKey(0)
			cv2.destroyAllWindows()


