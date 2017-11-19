import numpy as np
import cv2
import math

class CompletedLocalBinaryPattern:

    def __init__(self,R,P,mapping):
        self.radius = R
        self.numPoints = P
        self.mapping = mapping

    def get_pixel_else_0(self,l, idx, idy):
        if idx < int(len(l)) - 1 and idy < len(l[0]):
            return l[idx,idy]
        else:
            return 0

    def bilinear_interpolation(self,x, y, img):
        x1, y1 = int(x), int(y)
        x2, y2 = math.ceil(x), math.ceil(y)

        r1 = (x2 - x) / (x2 - x1) * self.get_pixel_else_0(img, x1, y1) + (x - x1) / (x2 - x1) * self.get_pixel_else_0(img, x2, y1)
        r2 = (x2 - x) / (x2 - x1) * self.get_pixel_else_0(img, x1, y2) + (x - x1) / (x2 - x1) * self.get_pixel_else_0(img, x2, y2)

        return (y2 - y) / (y2 - y1) * r1 + (y - y1) / (y2 - y1) * r2    

    def thresholded(self,center, pixels):
        out = []
        if len(pixels) > 0:
            out = np.where((pixels-center) >= 0,1,0)
        return out

    def getNeighboringPixels(self,img,R,P,x,y):
        pixels = []

        indexes = np.array(list(range(0,P)),dtype=np.float)
        dy = R * np.cos(2 * np.pi * indexes / P)
        dx = -R * np.sin(2 * np.pi * indexes / P)

        dx = np.where(abs(dx) < 5.0e-10, 0, dx)
        dy = np.where(abs(dy) < 5.0e-10, 0, dy)

        for point in range(0, P):
            r = x + dx[point]
            c = y + dy[point]

            if r < 0 or c < 0:
                pixels.append(0)
                continue            
            if int(r) == r:
                if int(c) != c:
                    c1 = int(c)
                    c2 = math.ceil(c)
                    w1 = (c2 - c) / (c2 - c1)
                    w2 = (c - c1) / (c2 - c1)
                                    
                    pixels.append(int((w1 * self.get_pixel_else_0(img, int(r), int(c)) + \
                                   w2 * self.get_pixel_else_0(img, int(r), math.ceil(c))) / (w1 + w2)))
                else:
                    pixels.append(self.get_pixel_else_0(img, int(r), int(c)))
            elif int(c) == c:
                r1 = int(r)
                r2 = math.ceil(r)
                w1 = (r2 - r) / (r2 - r1)
                w2 = (r - r1) / (r2 - r1)                
                pixels.append((w1 * self.get_pixel_else_0(img, int(r), int(c)) + \
                               w2 * self.get_pixel_else_0(img, math.ceil(r), int(c))) / (w1 + w2))
            else:
                pixels.append(self.bilinear_interpolation(r, c, img))

        return pixels


    def LBP(self,img,R,P):
        lbpImg = img.copy()

        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                center = img[x,y]
                pixels = self.getNeighboringPixels(img,R,P,x,y)                    
                values = self.thresholded(center, pixels)

                res = 0
                for a in range(0, len(values)):
                    res += values[a] * (2 ** a)
                lbpImg.itemset((x,y), res)

        return lbpImg


    def ULBP(self,img,R,P):
        uLbpImg = img.copy()
        p = np.array(list(range(1,P)))
        
        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                center = img[x,y]
                pixels = self.getNeighboringPixels(img,R,P,x,y)                    
                values = np.array(thresholded(center, pixels))

                res = abs(values[P-1] - values[0])
                res += np.sum(abs(values[p]-values[p-1]))
                uLbpImg.itemset((x,y), res)
                    
        return uLbpImg


    def LBPriu2(self,img,P,R):
        uLbpImg = ULBP(img,R,P)
        lbpRiu2Img = img.copy()

        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                if uLbpImg[x,y] <= 2:
                    center = img[x,y]
                    pixels = self.getNeighboringPixels(img,R,P,x,y)                    
                    values = np.array(thresholded(center, pixels))
                    lbpRiu2Img.itemset((x,y), np.sum(values))
                else:
                    lbpRiu2Img.itemset((x,y), P+1)
        return lbpRiu2Img


    def calcCLBPMThreshold(self,img,P,R):
        threshold = 0
        total = 0

        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                center = img[x,y]
                pixels = self.getNeighboringPixels(img,R,P,x,y)                    
                sp, mp = LDSMT(center, pixels)
                total += np.mean(mp)

        return total/(len(img)*len(img[0]))

    def calcLocalDifferences(self,img,P,R):
        ld = np.zeros((img.shape[0],img.shape[1],P),dtype=np.float)

        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                center = img[x,y]
                pixels = self.getNeighboringPixels(img,R,P,x,y)              
                ld[x,y,0:P] = pixels-center

        return ld

    def calcTransitions(self,pattern,P):
        u_value = np.absolute(pattern[:,:,P-1]-pattern[:,:,0])
        u_value += np.sum(np.absolute(pattern[:,:,1:P]-pattern[:,:,0:P-1]),2)
        return u_value

    def LDSMT(self,ld):
        sp = np.where(ld >= 0,1,-1)
        mp = np.absolute(ld)
        return sp, mp

    def CLBP_S(self,sp,P):
        sp = np.where(sp >= 0, 1, 0)
        pp2 = 2**(np.array(list(range(0,P))))
        return np.dot(sp,pp2)

    def CLBP_M(self,mp,P):
        c = np.mean(mp)
        tp = np.where(mp >= c, 1, 0)
        pp2 = (np.array(list(range(0,P))))
        return np.dot(tp,pp2)

    def CLBP_C(self,im):
        c = np.mean(im)
        return np.where(im >= c, 1, 0)

    def CLBP_S_riu2(self,sp,P):
        sp = np.where(sp >= 0, 1, 0)
        pp2 = np.array([1]*P)
        sp_total = np.dot(sp,pp2)
        u_value = self.calcTransitions(sp,P)
        return np.where(u_value <= 2, sp_total, P+1)

    def CLBP_M_riu2(self,mp,P):
        c = np.mean(mp)
        tp = np.where(mp >= c, 1, 0)
        pp2 = np.array([1]*P)
        tp_total = np.dot(tp,pp2.T)
        u_value = self.calcTransitions(tp,P)
        return np.where(u_value <= 2, tp_total, P+1)

    def genLocalPatterns(self,img):
        dp = self.calcLocalDifferences(img,self.numPoints,self.radius)
        sp, mp = self.LDSMT(dp)
        lpImg = []
        # "LBP" | "CLBP_S"
        if (self.mapping == "lbp") or (self.mapping == "clbp_s"):
            lpImg = self.CLBP_S(sp,self.numPoints)
        # "CLBP_S_riu2"
        elif self.mapping == "clbp_s_riu2":              
            lpImg = self.CLBP_S_riu2(sp,self.numPoints)
        # "CLBP_M"
        elif self.mapping == "clbp_m":               
            lpImg = self.CLBP_M(mp,self.numPoints)
        # "CLBP_M_riu2"
        elif self.mapping == "clbp_m_riu2":               
            lpImg = self.CLBP_M_riu2(mp,self.numPoints)
        # "CLBP_M_riu2"
        elif self.mapping == "clbp_c":               
            lpImg = self.CLBP_C(mp,self.numPoints)
        return lpImg

    def describe(self,img):
        dp = self.calcLocalDifferences(img,self.numPoints,self.radius)
        sp, mp = self.LDSMT(dp)

        # "LBP"
        if self.mapping == "lbp":                    
            lbp = self.CLBP_S(sp,self.numPoints)
            bins = list(range(0,2**self.numPoints))
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 2**self.numPoints))
        # "CLBP_S_riu2"
        elif self.mapping == "clbp_s":              
            clbp_s_riu2 = self.CLBP_S_riu2(sp,self.numPoints)
            hist = clbp_s_riu2.flatten()
        # "CLBP_M_riu2"
        elif self.mapping == "clbp_m":               
            clbp_m_riu2 = self.CLBP_M_riu2(mp,self.numPoints)
            hist = clbp_m_riu2.flatten()
        # "CLBP_M_riu2/C"
        elif self.mapping == "clbp_m/c":             
            clbp_m_riu2 = self.self.CLBP_M_riu2(mp,self.numPoints)
            clbp_c = self.CLBP_C(im)
            edges = list(range(0,2**P))
            hist2d, xedges, yedges = np.histogram2d(clbp_m_riu2.flatten(), 
                clbp_c.flatten(), 
                bins=(edges, edges))
            hist = hist2d.flatten()
        # "CLBP_S_riu2_M_riu2/C"
        elif self.mapping == "clbp_s_m/c":           
            clbp_m_riu2 = self.CLBP_M_riu2(mp,self.numPoints)
            clbp_c = self.CLBP_C(im)
            edges = list(range(0,2**P))
            clbp_m_riu2_c, xedges, yedges = np.histogram2d(clbp_m_riu2.flatten(), 
                clbp_c.flatten(), 
                bins=(edges, edges))
            clbp_s_riu2 = self.CLBP_S_riu2(sp,self.numPoints)
            hist = np.concatenate(clbp_s_riu2.flatten(),
                clbp_m_riu2_c.flatten(), 
                axis=0)
        # "CLBP_S_riu2/M_riu2"
        elif self.mapping == "clbp_s/m":             
            clbp_s_riu2 = self.CLBP_S_riu2(sp,self.numPoints)
            clbp_m_riu2 = self.CLBP_M_riu2(mp,self.numPoints)
            edges = list(range(0,2**P))
            hist2d, xedges, yedges = np.histogram2d(clbp_s_riu2.flatten(), 
                clbp_m_riu2.flatten(), 
                bins=(edges, edges))
            hist = hist2d.flatten()
        # "CLBP_S_riu2/M_riu2/C"
        elif self.mapping == "clbp_s/m/c":           
            clbp_s_riu2 = self.CLBP_S_riu2(sp,self.numPoints)
            feat1 = clbp_s_riu2.flatten()
            feat1.shape = (len(feat1),1)
            clbp_m_riu2 = self.CLBP_M_riu2(mp,self.numPoints)
            feat2 = clbp_m_riu2.flatten()
            feat2.shape = (len(feat2),1)
            clbp_c = self.CLBP_C(im)
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
