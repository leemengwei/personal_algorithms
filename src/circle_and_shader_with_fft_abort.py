import numpy as np
import matplotlib.pyplot as plt
import os,sys,time
import pandas as pd
from IPython import embed
import cv2
from matplotlib.patches import Circle

def get_data(data_path):
    data = cv2.imread(data_path)
    return data

def get_affine(data):
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    data = cv2.warpAffine(data, M, (cols, rows))
    return data

def get_perspective(data):
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    data = cv2.warpPerspective(data, M, (cols, rows))
    return data

def get_filter(data, size=5):
    kernel = np.ones((size,size),np.float32)/(size**2)
    data = cv2.filter2D(data, -1, kernel)
    return data

def get_bifilter(data, diameter=9):
    #Blur while keep boundary
    #???? bifilter together with binary will given new vein
    data = cv2.bilateralFilter(data, diameter,75,75)
    return data

def get_open_close(data):
    kernel = np.ones((3,3),np.uint8)
    data = cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel)
    data = cv2.morphologyEx(data, cv2.MORPH_CLOSE, kernel)
    #data = cv2.dilate(data, kernel,iterations = 1)
    #data = cv2.erode(data, kernel, iterations = 5)
    return data

def get_adaptive_binary(data):
    block_size = int(2*int(rows/200)+1)  
    #TODO: size should depend on physical size, then it will blender AND retain details of block.
    #Very big block: almost split whole pile.
    #Very small block: keeps every noise details.
    constant = 1
    gray_data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    data = cv2.adaptiveThreshold(gray_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
    return gray_data, data

def get_gradient(data):
    data = np.abs(cv2.Laplacian(data, cv2.CV_64F))
    data = np.clip(data, 0, 255)
    data = np.uint8(data)
    return data

def get_denoise(data):
    data = cv2.fastNlMeansDenoisingColored(data,None,3,1,7,7)
    return data

def cart2pol(x, y):
    radius = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(radius, phi)

def pol2cart(radius, phi):
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    return(x, y)

def get_x_y_on_a_cetain_circle(radius, anchor_point):
    x, y = anchor_point[0], anchor_point[1]
    phis = np.arange(0, 2*np.pi, 10/rows)
    rhos = np.tile(radius, len(phis))
    Xs, Ys = pol2cart(rhos, phis)
    Xs += x
    Ys += y
    Xs = np.clip(Xs, 0, cols-1)
    Ys = np.clip(Ys, 0, rows-1)
    return np.round(Xs).astype(int), np.round(Ys).astype(int), phis

def get_x_y_on_a_cetain_line(shader):
    _step = 1
    Xs = np.arange(0, cols, _step)     #TODO: _step given by physical
    Ys = np.tile(shader, len(Xs))
    #Xs = np.clip(Xs, 0, cols-1)
    #Ys = np.clip(Ys, 0, rows-1)
    return np.round(Xs).astype(int), np.round(Ys).astype(int), np.round(Xs).astype(int)

def get_fft(signal):
    # Frequency domain representation
    samplingFrequency   = 1000
    fourierTransform = np.fft.fft(signal)/len(signal)           # Normalize signal
    fourierTransform = fourierTransform[range(int(len(signal)/2))] # Exclude sampling frequency
    
    tpCount     = len(signal)
    values      = np.arange(int(tpCount/2))
    timePeriod  = tpCount/samplingFrequency
    frequencies = values/timePeriod
    idx = np.where(frequencies<=100)
    frequencies = frequencies[idx]
    fourierTransform = fourierTransform[idx]
    return frequencies, abs(fourierTransform)

def get_fft2d(img, size=90):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # 这里构建振幅图的公式没学过
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    fshift[int(crow-size):int(crow+size), int(ccol-size):int(ccol+size)] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back


if __name__ == "__main__":
    print("start...")

    data_name = sys.argv[1] #"1_env.jpg"

    raw_data = get_data('../data/'+data_name)
    rows, cols, ch = raw_data.shape

    #Transforms:
    #data = get_affine(data)
    #data = get_perspective(data)

    #data = get_bifilter(raw_data)
    #data = get_denoise(raw_data)
    gray_data, data = get_adaptive_binary(raw_data)
    data = data
   
    #some Anchors pre-got:
    anchor_num = 1
    #tmp_anchors = np.round(np.meshgrid(np.linspace(500, 3000, anchor_num), np.linspace(500,3000, anchor_num)))
    tmp_anchors = np.round(np.meshgrid(np.linspace(2000, 2000, anchor_num), np.linspace(1800, 1800, anchor_num)))
    anchors = np.array([tmp_anchors[0].flatten(), tmp_anchors[1].flatten()]).T

    #Get circle x y:
    for anchor_point in anchors:
        #for radius in np.arange(500, 1400, 10):
        history_fft_Y = np.array([])
        slide_step = 30
        for shader in np.arange(0, rows, slide_step):   #TODO: slide step given by physical
            plt.clf()
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(223)
            ax3 = plt.subplot(224)
            #Xs, Ys, pos = get_x_y_on_a_cetain_circle(radius, anchor_point)
            Xs, Ys, pos = get_x_y_on_a_cetain_line(shader)
            #embed()
            ax1.imshow(data)
            ax1.scatter(*anchors.T, s=2)
            ax1.scatter(Xs, Ys, s=3, color='r')
            try:
                pars = np.polyfit(pos, data[Ys, Xs], 10)
                trend = np.polyval(pars, pos)
                ax2.plot(data[Ys, Xs], label = 'raw')
                ax2.plot(trend, '--', label='trend')
                ax2.plot(data[Ys, Xs]-trend, label='no trend')
                #embed()
                fft_X, fft_Y = get_fft(data[Ys, Xs]-trend)
                history_fft_Y = np.append(history_fft_Y, fft_Y)
                ax3.plot(fft_X, fft_Y, label='fft of no trend')
                plt.legend()
            except Exception as e:
                print(e) 
            ax2.set_xticks(np.linspace(0, len(Xs), 10))
            ax2.set_xticklabels(np.linspace(0,360, 10))
            ax2.set_ylim(-150, 255)
            ax3.set_ylim(-5,5)
            plt.draw()
            plt.pause(0.01)
        realm_fft = history_fft_Y.reshape(-1, len(fft_Y)).sum(axis=0)/rows
        np.savetxt("realm_fft_%s"%data_name, np.vstack((fft_X, realm_fft)))
        plt.figure()
        plt.plot(fft_X, realm_fft)
        plt.show()
        



