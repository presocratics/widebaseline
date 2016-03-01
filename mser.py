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

import scipy
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

def getMask(img,con):
    h,w,c=img.shape
    mask=np.zeros((h,w),dtype=np.uint8)
    cv2.fillConvexPoly(mask,con,255)
    return mask

def getPatch(img,con):
    """Returns a patch from the contour"""
    mask=np.zeros(img.shape,dtype=img.dtype)
    cv2.fillConvexPoly(mask,con,(255,255,255))
    mask&=img
    x,y,w,h=cv2.boundingRect(con)
    return mask[y:y+h,x:x+w]

def handleInput():
    parser = argparse.ArgumentParser(description='Match reflections.')
    parser.add_argument('-f', type=float, default=800., help='Focal length.')
    parser.add_argument('-v', type=float, default=600., help='v0 center pixel in y')
    parser.add_argument('-t', type=float, default=2., help='Height above water in [m]')
    parser.add_argument('-n', type=float, default=5., help='Nearest distance to observe.')
    parser.add_argument('img', help='Image to search')
    args=parser.parse_args()
    return args

def findMSERCandidates(img):
    """Returns MSER candidates"""
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
    #regions = mser.detect(cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB),None)
    regions = mser.detect(img,None)
    return regions

def getSearchRegion(f,v0,t,mindist,mds,mdr,mean,hmax,hmin,isref=1,w=None):
    """isref: 1 is src, 2 is ref, 3 is both"""
    sr=[]
    max_disparity = 2.0*f*t/mindist
    if w is None:
        l=int(mean[0,0]-0.01*f)
        r=int(mean[0,0]+0.01*f)
    else:
        l=int(mean[0,0]-w/2)
        r=int(mean[0,0]+w/2)
    if isref==1 or isref==3: # source
        searchRegion=np.zeros((4,1,2),dtype=int)
        y=v0-mean[0,1]
        searchRegion[0,0,:] = (l,int(max(v0+y+mds,hmax[0,1])))
        searchRegion[1,0,:] = (r,int(max(v0+y+mds,hmax[0,1])))
        searchRegion[2,0,:] = (r,int(v0+y+max_disparity))
        searchRegion[3,0,:] = (l,int(v0+y+max_disparity))
        sr.append(searchRegion.copy())
    if isref==2 or isref==3: # reflection
        searchRegion=np.zeros((4,1,2),dtype=int)
        y=mean[0,1]-v0
        searchRegion[0,0,:] = (l, int(v0-y+mdr))
        searchRegion[1,0,:] = (r, int(v0-y+mdr))
        searchRegion[2,0,:] = (r, int(min(v0-y+max_disparity,hmin[0,1])))
        searchRegion[3,0,:] = (l, int(min(v0-y+max_disparity,hmin[0,1])))
        sr.append(searchRegion.copy())

    return sr


if __name__ == '__main__':
    args=handleInput()

    img = cv2.imread(args.img)
    h,w,c = img.shape
    vis = img.copy()
    focal_length=args.f

    """Calculate the maximum distance"""
    max_distance = np.sqrt(2.0*focal_length*args.t)
    max_disparity = 2.0*focal_length*args.t/args.n

    reflection_boundary1=args.v + focal_length*args.t/max_distance
    reflection_boundary2=args.v + focal_length*args.t/args.n

    regions = findMSERCandidates(img)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    cv2.drawContours(vis,hulls,-1,(255,0,0),1)
    cv2.line(vis, (0,int(args.v)), (w,int(args.v)), (0,255,0))
    cv2.line(vis, (0,int(reflection_boundary1)), (w,int(reflection_boundary1)), (0,0,255))
    cv2.line(vis, (0,int(reflection_boundary2)), (w,int(reflection_boundary2)), (0,0,255))
    cv2.imshow("reflection", vis)
    cv2.waitKey(0)

    for h in hulls:
        min_disparity_ref = 2.0*focal_length*args.t/max_distance
        min_disparity_src = 2.0*focal_length*args.t/max_distance
        hmax=h.max(axis=0)
        hmin=h.min(axis=0)
        hmean=h.mean(axis=0)
        hx,hy,hwidth,hheight = cv2.boundingRect(h)

        isref=1 # 1 is src, 2 is ref, 3 is both
        if hmean[0,1]>reflection_boundary2: # Must be a reflection
            isref=2
        elif hmean[0,1]>reflection_boundary1:
            isref=3
            min_disparity_src=2*(hmean[0,1]-args.v)
        searchRegions = getSearchRegion(args.f, args.v, args.t, args.n, 
                                        min_disparity_src,min_disparity_ref,
                                        hmean, hmax, hmin, isref,w=hwidth)

        vis=img.copy()

        """Construct search region"""
        '''
        good=[]
        for candidate in hulls:
            cmean=candidate.mean(axis=0)
            for sr in searchRegions:
                if cv2.pointPolygonTest(sr,(cmean[0,0],cmean[0,1]),False) < 0:
                    continue
                mask=getMask(img,sr)
                mkpt = srf.detect(img,mask)
                mk,md = descriptor.compute(img,mkpt)
                matches = matcher.match(sd, md)
                dist = [m.distance for m in matches]
                if len(dist)>0:
                    print("min: %f mean: %f max: %f" % (np.min(dist), np.mean(dist), np.max(dist)))
                    thresh_dist = (sum(dist) / len(dist)) * 0.5
                    sel_matches = [m for m in matches if m.distance < thresh_dist]
                    for m in sel_matches:
                        color = tuple([scipy.random.randint(0, 255) for _ in xrange(3)])
                        cv2.line(vis, (int(sk[m.queryIdx].pt[0]), int(sk[m.queryIdx].pt[1])) , (int(mk[m.trainIdx].pt[0]), int(mk[m.trainIdx].pt[1])), color)
                        print(sk[m.queryIdx].pt,mk[m.trainIdx].pt)
                good.append(candidate)

        #if len(good)==0:
        #    continue
        cv2.drawContours(vis,good,-1,(128,128,0),1)
        '''
        for sr in searchRegions:
            cv2.rectangle(vis,(sr[0,0,0],sr[0,0,1]),
                          (sr[2,0,0],sr[2,0,1]), (255,96,196))
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


