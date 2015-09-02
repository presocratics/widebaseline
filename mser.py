#! /usr/bin/python2.7
'''
MSER detector demo
==================

Usage:
------
    mser.py img
  
Keys:
-----
    ESC   - exit

'''

import numpy as np
import cv2

def scaled(con,sc):
    """Scales a contour by sc"""
    m=cv2.moments(con)
    mu=[ m['m10']/m['m00'], m['m01']/m['m00'] ]
    return sc*(con-mu)+mu

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
    import sys
    try: img_name = sys.argv[1]
    except: img_name = 0

    img = cv2.imread(img_name)
    mser = cv2.MSER(_min_area=15*15,_max_area=64*48)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    
    regions = mser.detect(gray, None)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    hulls2 = [scaled(p,2) for p in hulls]
    normedPatches=[normed(img,p,(100,100)) for p in hulls]
    cv2.namedWindow('src',cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('src',100,100)
    cv2.namedWindow('refl',cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('refl',300,100)
    cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('img',100,300)
    h,w,ch=img.shape
    for i in range(len(hulls)):
        mSrc=cv2.moments(hulls[i])
        muSrc=np.array([ mSrc['m10']/mSrc['m00'], mSrc['m01']/mSrc['m00'] ])

        src=normedPatches[i]
        srcFlipped=cv2.flip(src,0)
        maxScore=0
        maxIdx=-1
        for j in range(len(hulls)):
            if i!=j:
                mRef=cv2.moments(hulls[j])
                muRef=np.array([ mRef['m10']/mRef['m00'],
                                mRef['m01']/mRef['m00'] ])
                d=muRef-muSrc
                if abs(d[0])<6 and abs(d[1])>10:
                    corrRes=cv2.matchTemplate(srcFlipped,normedPatches[j],cv2.TM_CCORR_NORMED)
                    score=corrRes.max()
                    if score>maxScore:
                        maxIdx=j
                        maxScore=score
        if maxIdx!=-1 and maxScore>0.8:
            print maxScore,maxIdx
            cv2.imshow('src',srcFlipped)
            cv2.imshow('refl',normedPatches[maxIdx])
            cv2.drawContours(vis,hulls,i,(255,255,0),2)
            cv2.drawContours(vis,hulls,maxIdx,(255,255,0),2)
    cv2.imshow('img',vis)
    cv2.waitKey(0)

    #cv2.polylines(vis, hulls, 1, (0, 255, 0))
    #cv2.polylines(vis, [np.array(p,dtype=int) for p in hulls2], 1, (255, 0, 0))

    #cv2.imshow('img', vis)
    #cv2.waitKey(0)
    cv2.destroyAllWindows() 			
