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
import argparse

#depth function and cam calibration and collect images
#kuda functions in opencv


#def refl_thresh(focal_length, cam_height, distance_away, height_object):
#    theta = 2*np.arctan(.5*(cam_height/focal_length))
#    height_image = np.arctan(theta)*distance_away -height_object
#    return height_image

def disparity_thresh(cam_height, focal_length, image_height, neg_src_height): #returns max and min depth for a reflection in pixels
    max_distance = (2*focal_length*cam_height) #max distance to see one pixel clearly
    min_disparity = neg_src_height+(max_distance/(max_distance**.5))
    max_disparity = min_disparity+(max_distance/5)
    if max_disparity >    image_height: #if max_depth is greater than image height
        max_disparity = image_height #set max_depth = to image_height
    return min_disparity, max_disparity

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
    parser = argparse.ArgumentParser(description='Match reflections.')
    parser.add_argument('-f', type=float, default=800., help='Focal length.')
    parser.add_argument('-v', type=float, default=600., help='v0 center pixel in y')
    parser.add_argument('-t', type=float, default=2., help='Height above water in [m]')
    parser.add_argument('-n', type=float, default=5., help='Nearest distance to observe.')
    parser.add_argument('img', help='Image to search')
    args=parser.parse_args()

    img = cv2.imread(args.img)
    focal_length=args.f

    """Calculate the maximum distance"""
    max_distance = np.sqrt(2.0*focal_length*args.t)
    max_disparity = 2.0*focal_length*args.t/args.n

    reflection_boundary1=args.v + focal_length*args.t/max_distance
    reflection_boundary2=args.v + focal_length*args.t/args.n

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    h,w,ch=img.shape
    mina = 15*15 # minarea default: 60
    maxa = h*w/100 # maxarea default: 14400
    mv = 0.25 # max variation (to do with child size?) default: 0.25
    md = 0.2 # min diversity (for color images, what is diversity?) default: 0.2
    me = 200 # max evolution (for color images, num steps) default: 200
    at = 1.01 # area threshold (for color, area threshold that causes re-init?) default: 1.01
    mm = 0.003 # min margin (for color, ignore too small margin?) default: 0.003
    ebs = 5 # edge blur size (for color, aperture (kernel?) size for edge blur) default: 5

    mser = cv2.MSER(_min_area=15*15,_max_area=h*w/100, _max_variation=mv,
                    _min_diversity=md)
    regions = mser.detect(img, None)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.drawContours(vis,hulls,-1,(255,0,0),1)
    cv2.line(vis, (0,int(args.v)), (w,int(args.v)), (0,255,0))
    cv2.line(vis, (0,int(reflection_boundary1)), (w,int(reflection_boundary1)), (0,0,255))
    cv2.line(vis, (0,int(reflection_boundary2)), (w,int(reflection_boundary2)), (0,0,255))
    cv2.imshow("reflection", vis)
    cv2.waitKey(0)
    for h in hulls:
        min_disparity = 2.0*focal_length*args.t/max_distance
        hmax=h.max(axis=0)
        if hmax[0,1]>reflection_boundary2: # cannot be a source
            continue
        if hmax[0,1]>reflection_boundary1:
            print(max_distance,(focal_length*args.t)/(hmax[0,1]-args.v))
            min_disparity=2*(hmax[0,1]-args.v)
        vis=img.copy()
        harea=cv2.contourArea(h)
        hmean=h.mean(axis=0)
        y=args.v-hmean[0,1]
        hx,hy,hwidth,hheight = cv2.boundingRect(h)
        """Construct search region"""
        searchRegion=np.zeros((4,1,2),dtype=int)
        searchRegion[0,0,:] = (int(hmean[0,0]-0.01*focal_length),int(max(args.v+y+min_disparity,hmean[0,1])))
        searchRegion[1,0,:] = (int(hmean[0,0]+0.01*focal_length),int(max(args.v+y+min_disparity,hmean[0,1])))
        searchRegion[2,0,:] = (int(hmean[0,0]+0.01*focal_length),int(args.v+y+max_disparity))
        searchRegion[3,0,:] = (int(hmean[0,0]-0.01*focal_length),int(args.v+y+max_disparity))

        good=[]
        for candidate in hulls:
            cmean=candidate.mean(axis=0)
            if cv2.pointPolygonTest(searchRegion,(cmean[0,0],cmean[0,1]),False) < 0:
                continue
            good.append(candidate)

        if len(good)==0:
            continue
        cv2.drawContours(vis,good,-1,(128,128,0),1)
        cv2.rectangle(vis,(searchRegion[0,0,0],searchRegion[0,0,1]),
                      (searchRegion[2,0,0],searchRegion[2,0,1]), (255,96,196))
        cv2.drawContours(vis,[h],-1,(255,0,0),1)
        cv2.line(vis, (0,int(reflection_boundary1)), (w,int(reflection_boundary1)), (0,0,255))
        cv2.line(vis, (0,int(reflection_boundary2)), (w,int(reflection_boundary2)), (0,0,255))
        cv2.line(vis, (0,int(args.v)), (w,int(args.v)), (0,255,0))
        cv2.imshow("reflection", vis)
        cv2.waitKey(0)

    '''
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
                if abs(d[0])<6 and abs(d[1])>8: 
                    width_upperbound = dim2*1.1 #width must be within +/- 10%
                    width_lowerbound = dim2*.9
                    if dim1[0,0] > width_lowerbound[0,0] and dim1[0,0] < width_upperbound[0,0]:
                        height_lowerbound = (2*horizon_line)- muSrc[1] # lower bound distance on how close reflection is to horizon_line
                        if muRef[1] > height_lowerbound:
                            neg_src_height = 2*horizon_line - muSrc[1]
                            min_disparity, max_disparity = disparity_thresh(2, focal_length, h, neg_src_height) #obtain min and max depth thresholds for the reflections
                            print(max_disparity, min_disparity)
                            if muRef[1] < max_disparity and muRef[1] > min_disparity:
                                corrRes=cv2.matchTemplate(srcFlipped,normedPatches[j],cv2.TM_CCOEFF) #TM_CCOEFF are effective methods
                                score=corrRes.max()
                                if score>maxScore:
                                    maxIdx=j
                                    maxScore=score
        if maxIdx!=-1 and maxScore>0.9:
            color = [np.random.uniform(150,255), np.random.uniform(150,255), 255]            
            cv2.drawContours(vis,source,i,color,2)
            cv2.drawContours(vis,reflect,maxIdx,color,2)
            #cv2.namedWindow("reflection1.png", cv2.WINDOW_NORMAL)
            cv2.imshow("reflection1.png", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    '''


