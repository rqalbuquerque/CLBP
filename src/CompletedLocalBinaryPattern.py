import numpy as np
import cv2
import math

def get_pixel_else_0(l, idx, idy):
    if idx < int(len(l)) - 1 and idy < len(l[0]):
        return l[idx,idy]
    else:
        return 0

def bilinearInterpolation(x, y, img):
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

def getNeighboringPixelsPaperVersion(img,R,P,x,y):
    pixels = []

    indexes = np.array(list(range(0,P)),dtype=np.float)
    dy = -R * np.sin(2 * np.pi * indexes / P)
    dx = R * np.cos(2 * np.pi * indexes / P)

    dy = np.where(abs(dy) < 5.0e-10, 0, dy)
    dx = np.where(abs(dx) < 5.0e-10, 0, dx)

    for point in range(0, P):
        r = y + dy[point]
        c = x + dx[point]

        fr = math.floor(r)
        fc = math.floor(c)

        cr = math.ceil(r)
        cc = math.ceil(c)
        
        rr = np.round(r)
        rc = np.round(c)

        if abs(c-rc) < 10e-7 and abs(r-rr) < 10e-7:
            pixels.append(get_pixel_else_0(img, int(r), int(c)))
        else:
            tr = r - fr
            tc = c - fc

            w1 = (1 - tc) * (1 - tr)
            w2 =      tc  * (1 - tr)
            w3 = (1 - tc) *      tr 
            w4 =      tc  *      tr 

            value = w1*get_pixel_else_0(img, fr, fc) + w2*get_pixel_else_0(img, fr, cc) + \
                    w3*get_pixel_else_0(img, cr, fc) + w4*get_pixel_else_0(img, cr, cc)

            pixels.append(value)

    return pixels


def getNeighboringPixels(img,R,P,x,y):
    pixels = []

    indexes = np.array(list(range(0,P)),dtype=np.float)
    dy = -R * np.sin(2 * np.pi * indexes / P)
    dx = R * np.cos(2 * np.pi * indexes / P)

    dy = np.where(abs(dy) < 5.0e-10, 0, dy)
    dx = np.where(abs(dx) < 5.0e-10, 0, dx)

    for point in range(0, P):
        c = x + dx[point]
        r = y + dy[point]

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
            pixels.append(bilinearInterpolation(r, c, img))

    return pixels

def genMappingTable(P):
    mapTable = np.zeros(2**P)
    numElms = P+2
    for i in range(0,2**P):
        msb = 128 & i
        lsb = 1 & i
        count = format(i, '#010b').count('01') + format(i, '#010b').count('10') + ((msb >> (P-1)) ^ lsb)

        if count <= 2:
            mapTable[i] = bin(i).count('1')
        else:
            mapTable[i] = P+1

    return mapTable

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
    return lbpRiu2Img


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

    for y in range(0, len(img)):
        for x in range(0, len(img[0])):
            center = img[y,x]
            #pixels = getNeighboringPixels(img,R,P,x,y)              
            pixels = getNeighboringPixelsPaperVersion(img,R,P,x,y)              
            ld[y,x,0:P] = pixels-center

    return ld

def calcTransitions(pattern,P):
    u_value = np.absolute(pattern[:,:,P-1]-pattern[:,:,0])
    u_value += np.sum(np.absolute(pattern[:,:,1:P]-pattern[:,:,0:P-1]),2)
    return u_value

def LDSMT(ld):
    sp = np.where(ld >= 0,1,-1)
    mp = np.absolute(ld)
    return sp, mp

def CLBP_S(sp,P):
    sp = np.where(sp >= 0, 1, 0)
    pp2 = 2**(np.array(list(range(0,P))))
    return np.dot(sp,pp2)

def CLBP_M(mp,P):
    c = np.mean(mp)
    tp = np.where(mp >= c, 1, 0)
    pp2 = (np.array(list(range(0,P))))
    return np.dot(tp,pp2)

def CLBP_C(im):
    c = np.mean(im)
    return np.where(im >= c, 1, 0)

def CLBP_S_riu2(sp,P):
    sp = np.where(sp >= 0, 1, 0)
    #sp_total = np.sum(sp,2)
    #u_value = calcTransitions(sp,P)
    #return np.where(u_value <= 2, sp_total, P+1)
    pp2 = 2**(np.array(list(range(0,P))))
    indexes = np.dot(sp,pp2)    
    return indexes

def CLBP_M_riu2(mp,P):
    c = np.mean(mp)
    tp = np.where(mp >= c, 1, 0)
    # pp2 = np.array([1]*P)
    # tp_total = np.dot(tp,pp2.T)
    # u_value = calcTransitions(tp,P)
    # return np.where(u_value <= 2, tp_total, P+1)
    pp2 = 2**(np.array(list(range(0,P))))
    indexes = np.dot(tp,pp2)    
    return indexes

def genLocalPatterns(img):
    dp = calcLocalDifferences(img,numPoints,radius)
    sp, mp = LDSMT(dp)
    lpImg = []
    # "LBP" | "CLBP_S"
    if (mapping == "lbp") or (mapping == "clbp_s"):
        lpImg = CLBP_S(sp,numPoints)
    # "CLBP_S_riu2"
    elif mapping == "clbp_s_riu2":              
        lpImg = CLBP_S_riu2(sp,numPoints)
    # "CLBP_M"
    elif mapping == "clbp_m":               
        lpImg = CLBP_M(mp,numPoints)
    # "CLBP_M_riu2"
    elif mapping == "clbp_m_riu2":               
        lpImg = CLBP_M_riu2(mp,numPoints)
    # "CLBP_M_riu2"
    elif mapping == "clbp_c":               
        lpImg = CLBP_C(mp,numPoints)
    return lpImg

def describeHistogram(img, sp, mp, mapping):
    # "LBP"
    if mapping == "lbp":                    
        lbp = CLBP_S(sp,numPoints)
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 2**numPoints))
    # "CLBP_S_riu2"
    elif mapping == "clbp_s":              
        clbp_s_riu2 = CLBP_S_riu2(sp,numPoints)
        (hist, bins) = np.histogram(clbp_s_riu2.ravel(), bins=np.arange(0, numPoints+3))
    # "CLBP_M_riu2"
    elif mapping == "clbp_m":               
        clbp_m_riu2 = CLBP_M_riu2(mp,numPoints)
        (hist, bins) = np.histogram(clbp_m_riu2.ravel(), bins=np.arange(0, numPoints+3))
    # "CLBP_M_riu2/C"
    elif mapping == "clbp_m/c":             
        clbp_m_riu2 = CLBP_M_riu2(mp,numPoints)
        clbp_c = CLBP_C(img)
        edgeM = list(range(0,numPoints+3))
        edgeC = list(range(0,3))
        hist2d, xedges, yedges = np.histogram2d(clbp_m_riu2.ravel(), 
            clbp_c.ravel(), 
            bins=(edgeM, edgeC))
        hist = hist2d.flatten()
    # "CLBP_S_riu2_M_riu2/C"
    elif mapping == "clbp_s_m/c":           
        clbp_m_riu2 = CLBP_M_riu2(mp,numPoints)
        clbp_c = CLBP_C(img)
        edges = list(range(0,2**P))
        clbp_m_riu2_c, xedges, yedges = np.histogram2d(clbp_m_riu2.flatten(), 
            clbp_c.flatten(), 
            bins=(edges, edges))
        clbp_s_riu2 = CLBP_S_riu2(sp,numPoints)
        hist = np.concatenate(clbp_s_riu2.flatten(),
            clbp_m_riu2_c.flatten(), 
            axis=0)
    # "CLBP_S_riu2/M_riu2"
    elif mapping == "clbp_s/m":             
        clbp_s_riu2 = CLBP_S_riu2(sp,numPoints)
        clbp_m_riu2 = CLBP_M_riu2(mp,numPoints)
        edges = list(range(0,2**P))
        hist2d, xedges, yedges = np.histogram2d(clbp_s_riu2.flatten(), 
            clbp_m_riu2.flatten(), 
            bins=(edges, edges))
        hist = hist2d.flatten()
    # "CLBP_S_riu2/M_riu2/C"
    elif mapping == "clbp_s/m/c":           
        clbp_s_riu2 = CLBP_S_riu2(sp,numPoints)
        feat1 = clbp_s_riu2.flatten()
        feat1.shape = (len(feat1),1)
        clbp_m_riu2 = CLBP_M_riu2(mp,numPoints)
        feat2 = clbp_m_riu2.flatten()
        feat2.shape = (len(feat2),1)
        clbp_c = CLBP_C(img)
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



def chiSquared(p,q):
    return np.sum((p-q)**2/(p+q+1e-6))


if __name__ == "__main__":

    database = "C:/Users/rqa/Desktop/CLBP/database/Outex-TC-00010/000/"

    radius = 1
    numPoints = 8

    clbp_s = []
    clbp_m = []
    clbp_mc = []
    clbp_s_mc = []
    clbp_sm = []
    clbp_smc = []

    labels = []
    
    result_labels_clbp_s = []
    result_labels_clbp_m = []
    result_labels_clbp_mc = []
    result_labels_clbp_s_mc = []
    result_labels_clbp_sm = []
    result_labels_clbp_smc = []

    # nn_classifier_clbp_s = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)
    # nn_classifier_clbp_m = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)
    # nn_classifier_clbp_mc = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)
    # nn_classifier_clbp_s_mc = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)
    # nn_classifier_clbp_sm = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)
    # nn_classifier_clbp_smc = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)

    mappingTable = genMappingTable(numPoints)

    # classes
    with open(database + "classes.txt", "r") as file:
        numClasses = int(file.readline())
        classes = {}
        for line in file:
            columns = line.split()
            classes[int(columns[1])] = columns[0]

    # train
    print("Training steps:")
    with open(database + "train_mini.txt", "r") as train:
        with open(database + "train_files.txt", "r") as trainFiles:
            numTrain = int(train.readline())
            i = 0
            for pathFile in trainFiles:
                img = cv2.imread(pathFile.rstrip(), cv2.IMREAD_GRAYSCALE)
                img = img/255
                img = (img-np.mean(img))/np.std(img)*20+128

                dp = calcLocalDifferences(img,numPoints,radius)
                sp, mp = LDSMT(dp)

                # CLBP_C
                clbp_c = CLBP_C(img)

                # CLBP_S_riu2
                clbp_s_riu2 = CLBP_S_riu2(sp,numPoints)
                mapped_clbp_s_riu2 = mappingTable[clbp_s_riu2]
                (hist_s, bins) = np.histogram(mapped_clbp_s_riu2.ravel(1), bins=numPoints+2)
                clbp_s.append(hist_s.ravel(1))
                
                # CLBP_M_riu2
                clbp_m_riu2 = CLBP_M_riu2(mp,numPoints)
                mapped_clbp_m_riu2 = mappingTable[clbp_m_riu2]
                (hist_m, bins) = np.histogram(mapped_clbp_m_riu2.ravel(1), bins=numPoints+2)
                clbp_m.append(hist_m.ravel(1))

                # CLBP_M/C
                hist_mc, xedges, yedges = np.histogram2d(mapped_clbp_m_riu2.ravel(1), clbp_c.ravel(1),bins=[numPoints+2,2])
                clbp_mc.append(hist_mc.ravel(1))

                # CLBP_S_M/C
                hist_s_mc = np.concatenate((hist_s,hist_mc.ravel(1)),axis=0)
                clbp_s_mc.append(hist_s_mc.ravel(1))

                # CLBP_S/M
                hist_sm, xedges, yedges = np.histogram2d(mapped_clbp_s_riu2.ravel(1),mapped_clbp_m_riu2.ravel(1),bins=[numPoints+2,numPoints+2])
                clbp_sm.append(hist_sm.ravel(1))

                # CLBP_S/M/C
                clbp_mc_sum = np.where(clbp_c > 0, clbp_m_riu2+numPoints+2, clbp_m_riu2)
                hist_smc, xedges, yedges = np.histogram2d(mapped_clbp_s_riu2.ravel(1), clbp_mc_sum.ravel(1),bins=[numPoints+2,2*(numPoints+2)])
                clbp_smc.append(hist_smc.ravel(1))

                labels.append(train.readline().split()[1])

                print(i)
                i += 1
                if i>=numTrain:
                    break
                    
    # fitting
    print("Fitting the models")

    # model = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto', metric=chiSquared)
    # model.fit(data,labels)

    # nn_classifier_clbp_s.fit(clbp_s,labels)
    # nn_classifier_clbp_m.fit(clbp_m,labels)
    # nn_classifier_clbp_mc.fit(clbp_mc,labels)
    # nn_classifier_clbp_s_mc.fit(clbp_s_mc,labels)
    # nn_classifier_clbp_sm.fit(clbp_sm,labels)
    # nn_classifier_clbp_smc.fit(clbp_smc,labels)

    # test
    print("Test steps:")
    with open(database + "test_mini.txt", "r") as test:
        with open(database + "test_files.txt", "r") as testFiles:
            numTest = int(test.readline())
            i = 0
            for pathFile in testFiles:
                img = cv2.imread(pathFile.rstrip(), cv2.IMREAD_GRAYSCALE)
                img = img/255
                img = (img-np.mean(img))/np.std(img)*20+128

                dp = calcLocalDifferences(img,numPoints,radius)
                sp, mp = LDSMT(dp)

                # CLBP_C
                clbp_c = CLBP_C(img)

                # CLBP_S_riu2
                clbp_s_riu2 = CLBP_S_riu2(sp,numPoints)
                mapped_clbp_s_riu2 = mappingTable[clbp_s_riu2]
                (hist_s, bins) = np.histogram(mapped_clbp_s_riu2.ravel(1), bins=numPoints+2)
                # result_labels_clbp_s.append((test.readline().split()[1],
                #     nn_classifier_clbp_s.predict(hist_s.ravel(1))))
                
                # CLBP_M_riu2
                clbp_m_riu2 = CLBP_M_riu2(mp,numPoints)
                mapped_clbp_m_riu2 = mappingTable[clbp_m_riu2]
                (hist_m, bins) = np.histogram(mapped_clbp_m_riu2.ravel(1), bins=numPoints+2)
                # result_labels_clbp_m.append((test.readline().split()[1],
                #   nn_classifier_clbp_m.predict(hist_m.ravel(1))))


                # CLBP_M/C
                hist_mc, xedges, yedges = np.histogram2d(mapped_clbp_m_riu2.ravel(1), clbp_c.ravel(1),bins=[numPoints+2,2])
                # result_labels_clbp_mc.append((test.readline().split()[1],
                #   nn_classifier_clbp_mc.predict(hist_mc.ravel(1))))

                # CLBP_S_M/C
                hist_s_mc = np.concatenate((hist_s,hist_mc.ravel(1)),axis=0)
                # result_labels_clbp_s_mc.append((test.readline().split()[1],
                #   nn_classifier_clbp_s_mc.predict(hist_s_mc.ravel(1))))

                # CLBP_S/M
                hist_sm, xedges, yedges = np.histogram2d(mapped_clbp_s_riu2.ravel(1),mapped_clbp_m_riu2.ravel(1),bins=[numPoints+2,numPoints+2])
                # result_labels_clbp_sm.append((test.readline().split()[1],
                #   nn_classifier_clbp_sm.predict(hist_sm.ravel(1))))

                # CLBP_S/M/C
                clbp_mc_sum = np.where(clbp_c > 0, clbp_m_riu2+numPoints+2, clbp_m_riu2)
                hist_smc, xedges, yedges = np.histogram2d(mapped_clbp_s_riu2.ravel(1), clbp_mc_sum.ravel(1),bins=[numPoints+2,2*(numPoints+2)])
                # result_labels_clbp_smc.append((test.readline().split()[1],
                #   nn_classifier_clbp_smc.predict(hist_smc.ravel(1))))

                print(i)
                i += 1
                if i>=numTrain:
                    break

    # result_labels_clbp_s = np.array(result_labels_clbp_s)
    # acc_labels_clbp_s = (result_labels_clbp_s[:,0] == result_labels_clbp_s[:,1])/numTest

    # result_labels_clbp_m = np.array(result_labels_clbp_m)
    # acc_labels_clbp_m = (result_labels_clbp_m[:,0] == result_labels_clbp_m[:,1])/numTest

    # result_labels_clbp_mc = np.array(result_labels_clbp_mc)
    # acc_labels_clbp_mc = (result_labels_clbp_mc[:,0] == result_labels_clbp_mc[:,1])/numTest

    # result_labels_clbp_s_mc = np.array(result_labels_clbp_s_mc)
    # acc_labels_clbp_s_mc = (result_labels_clbp_s_mc[:,0] == result_labels_clbp_s_mc[:,1])/numTest

    # result_labels_clbp_sm = np.array(result_labels_clbp_sm)
    # acc_labels_clbp_sm = (result_labels_clbp_sm[:,0] == result_labels_clbp_sm[:,1])/numTest

    # result_labels_clbp_smc = np.array(result_labels_clbp_smc)
    # acc_labels_clbp_smc = (result_labels_clbp_smc[:,0] == result_labels_clbp_smc[:,1])/numTest

