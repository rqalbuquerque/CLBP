import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

class CompletedLocalBinaryPattern:

    def __init__(self,R,P,mapping):
        self.radius = R
        self.numPoints = P
        self.mapping = mapping

    def bilinear_interpolation(x, y, img):
        x1, y1 = int(x), int(y)
        x2, y2 = math.ceil(x), math.ceil(y)

        r1 = (x2 - x) / (x2 - x1) * get_pixel_else_0(img, x1, y1) + (x - x1) / (x2 - x1) * get_pixel_else_0(img, x2, y1)
        r2 = (x2 - x) / (x2 - x1) * get_pixel_else_0(img, x1, y2) + (x - x1) / (x2 - x1) * get_pixel_else_0(img, x2, y2)

        return (y2 - y) / (y2 - y1) * r1 + (y - y1) / (y2 - y1) * r2    


    def thresholded(center, pixels):
        out = []
        if len(pixels) > 0:
            out = np.where((pixels-center) >= 0,1,0)
        return out


    def get_pixel_else_0(l, idx, idy):
        if idx < int(len(l)) - 1 and idy < len(l[0]):
            return l[idx,idy]
        else:
            return 0


    def getNeighboringPixels(img,R,P,x,y):

        pixels = []

        for point in range(0, P):
            r = x + R * math.cos(2 * math.pi * point / P)
            c = y - R * math.sin(2 * math.pi * point / P)
            if r < 0 or c < 0:
                pixels.append(0)
                continue            
            if int(r) == r:
                if int(c) != c:
                    c1 = int(c)
                    c2 = math.ceil(c)
                    w1 = (c2 - c) / (c2 - c1)
                    w2 = (c - c1) / (c2 - c1)
                                    
                    pixels.append(int((w1 * get_pixel_else_0(img, int(r), int(c)) + \
                                   w2 * get_pixel_else_0(img, int(r), math.ceil(c))) / (w1 + w2)))
                else:
                    pixels.append(get_pixel_else_0(img, int(r), int(c)))
            elif int(c) == c:
                r1 = int(r)
                r2 = math.ceil(r)
                w1 = (r2 - r) / (r2 - r1)
                w2 = (r - r1) / (r2 - r1)                
                pixels.append((w1 * get_pixel_else_0(img, int(r), int(c)) + \
                               w2 * get_pixel_else_0(img, math.ceil(r), int(c))) / (w1 + w2))
            else:
                pixels.append(bilinear_interpolation(r, c, img))

        return pixels


    def LBP(img,R,P):

        lbpImg = img.copy()

        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                center = img[x,y]
                pixels = getNeighboringPixels(img,R,P,x,y)                    
                values = thresholded(center, pixels)

                res = 0
                for a in range(0, len(values)):
                    res += values[a] * (2 ** a)
                lbpImg.itemset((x,y), res)

        return lbpImg


    def ULBP(img,R,P):

        uLbpImg = img.copy()
        p = np.array(list(range(1,P)))
        
        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                center = img[x,y]
                pixels = getNeighboringPixels(img,R,P,x,y)                    
                values = np.array(thresholded(center, pixels))

                res = abs(values[P-1] - values[0])
                res += np.sum(abs(values[p]-values[p-1]))
                uLbpImg.itemset((x,y), res)
                    
        return uLbpImg


    def LBPriu2(img,P,R):

        uLbpImg = ULBP(img,R,P)
        lbpRiu2Img = img.copy()

        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                if uLbpImg[x,y] <= 2:
                    center = img[x,y]
                    pixels = getNeighboringPixels(img,R,P,x,y)                    
                    values = np.array(thresholded(center, pixels))
                    lbpRiu2Img.itemset((x,y), np.sum(values))
                else:
                    lbpRiu2Img.itemset((x,y), P+1)



    def calcCLBPMThreshold(img,P,R):

        threshold = 0
        total = 0

        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                center = img[x,y]
                pixels = getNeighboringPixels(img,R,P,x,y)                    
                sp, mp = LDSMT(center, pixels)
                total += np.mean(mp)

        return total/(len(img)*len(img[0]))


    def calcLocalDifferences(img,P,R):

        ld = np.zeros((img.shape[0],img.shape[1],P),dtype=np.float)

        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                center = img[x,y]
                pixels = getNeighboringPixels(img,R,P,x,y)                    
                ld[x,y,:] = pixels-center
        return ld

    def calcTransitions(pattern,P):
        u_value = np.absolute(pattern[:,:,P-1]-pattern[:,:,0])
        u_value += np.sum(np.absolute(pattern[:,:,1:P-1]-pattern[:,:,0:P-2]),2)
        return u_value

    def LDSMT(ld):
        sp = np.where(ld >= 0,1,-1)
        mp = np.absolute(ld)
        return sp, mp

    def CLBP_S(sp,P):
        sp = np.where(sp >= 0, 1, 0)
        pp2 = 2**(np.array(list(range(0,P))))
        return np.dot(sp,pp2.T)

    def CLBP_M(mp,P):
        c = np.mean(mp)
        tp = np.where(mp >= c, 1, 0)
        pp2 = (np.array(list(range(0,P))))
        return np.dot(tp,pp2.T)

    def CLBP_C(im):
        c = np.mean(im)
        return np.where(im >= c, 1, 0)

    def CLBP_S_riu2(sp,P):
        sp = np.where(sp >= 0, 1, 0)
        pp2 = np.array([1]*P)
        sp_total = np.dot(sp,pp2.T)
        u_value = calcTransitions(sp,P)
        return np.where(u_value <= 2, sp_total, P-1)

    def CLBP_M_riu2(mp,P):
        c = np.mean(mp)
        tp = np.where(mp >= c, 1, 0)
        pp2 = np.array([1]*P)
        tp_total = np.dot(tp,pp2.T)
        u_value = calcTransitions(tp,P)
        return np.where(u_value <= 2, tp_total, P-1)

    def describe(img):
        dp = calcLocalDifferences(img,self.numPoints,self.radius)
        sp, mp = LDSMT(dp)

        # "LBP"
        if self.mapping == "lbp":                    
            hist = CLBP_S(sp,self.numPoints)
            hist = lbp.flatten()
        # "CLBP_S_riu2"
        elif self.mapping == "clbp_s":              
            clbp_s_riu2 = CLBP_S_riu2(sp,self.numPoints)
            hist = clbp_s_riu2.flatten()
        # "CLBP_M_riu2"
        elif self.mapping == "clbp_m":               
            clbp_m_riu2 = CLBP_M_riu2(mp,self.numPoints)
            hist = clbp_m_riu2.flatten()
        # "CLBP_M_riu2/C"
        elif self.mapping == "clbp_m/c":             
            clbp_m_riu2 = CLBP_M_riu2(mp,self.numPoints)
            clbp_c = CLBP_C(im)
            edges = list(range(0,2**P))
            hist2d, xedges, yedges = np.histogram2d(clbp_m_riu2.flatten(), 
                clbp_c.flatten(), 
                bins=(edges, edges))
            hist = hist2d.flatten()
        # "CLBP_S_riu2_M_riu2/C"
        elif self.mapping == "clbp_s_m/c":           
            clbp_m_riu2 = CLBP_M_riu2(mp,self.numPoints)
            clbp_c = CLBP_C(im)
            edges = list(range(0,2**P))
            clbp_m_riu2_c, xedges, yedges = np.histogram2d(clbp_m_riu2.flatten(), 
                clbp_c.flatten(), 
                bins=(edges, edges))
            clbp_s_riu2 = CLBP_S_riu2(sp,self.numPoints)
            hist = np.concatenate(clbp_s_riu2.flatten(),
                clbp_m_riu2_c.flatten(), 
                axis=0)
        # "CLBP_S_riu2/M_riu2"
        elif self.mapping == "clbp_s/m":             
            clbp_s_riu2 = CLBP_S_riu2(sp,self.numPoints)
            clbp_m_riu2 = CLBP_M_riu2(mp,self.numPoints)
            edges = list(range(0,2**P))
            hist2d, xedges, yedges = np.histogram2d(clbp_s_riu2.flatten(), 
                clbp_m_riu2.flatten(), 
                bins=(edges, edges))
            hist = hist2d.flatten()
        # "CLBP_S_riu2/M_riu2/C"
        elif self.mapping == "clbp_s/m/c":           
            clbp_s_riu2 = CLBP_S_riu2(sp,self.numPoints)
            feat1 = clbp_s_riu2.flatten()
            feat1.shape = (len(feat1),1)
            clbp_m_riu2 = CLBP_M_riu2(mp,self.numPoints)
            feat2 = clbp_m_riu2.flatten()
            feat2.shape = (len(feat2),1)
            clbp_c = CLBP_C(im)
            feat3 = clbp_c.flatten()
            feat3.shape = (len(feat3),1)
            feats = np.concatenate((feat1, feat2, feat3), 
                axis=1)
            edges = list(range(0,2**P))
            hist3d, edges = np.histogramdd(r, 
                bins = (edges, edges, edges))
            hist = hist3d.flatten()

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + np.finfo(float).eps)

        return hist
