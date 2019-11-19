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

def get_filter(data):
    kernel = np.ones((5,5),np.float32)/25
    data = cv2.filter2D(data, -1, kernel)
    return data

def get_bifilter(data):
    #Blur while keep boundary
    data = cv2.bilateralFilter(data,9,75,75)
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
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def get_x_y_on_a_cetain_circle(rho, anchor):
    x, y = anchor[0], anchor[1]
    phis = np.arange(0, 2*np.pi, 10/rows)
    rhos = np.tile(rho, len(phis))
    Xs, Ys = pol2cart(rhos, phis)
    Xs += x-1
    Ys += y-1
    return np.round(Xs).astype(int), np.round(Ys).astype(int)

if __name__ == "__main__":
    print("start...")

    data_path = "../data/1.jpg"

    raw_data = get_data(data_path)
    rows, cols, ch = raw_data.shape

    #Transforms:
    #data = get_affine(data)
    #data = get_perspective(data)

    #data = get_bifilter(raw_data)
    #data = get_denoise(raw_data)
    gray_data, data = get_adaptive_binary(raw_data)
    data = gray_data
    
    #some Anchors pre-got:
    anchor_num = 4
    tmp_anchors = np.round(np.meshgrid(np.linspace(500, 3000, anchor_num), np.linspace(500,3000, anchor_num)))
    anchors = np.array([tmp_anchors[0].flatten(), tmp_anchors[1].flatten()]).T

    #Get circle x y:
    for anchor in anchors:
        for rho in np.arange(10, 600, 20):
            print("On r:", rho)
            plt.clf()
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)
            Xs, Ys = get_x_y_on_a_cetain_circle(rho, anchor)
            #embed()
            ax1.imshow(data, cmap='gray')
            ax1.scatter(*anchors.T, s=2)
            ax1.scatter(Xs, Ys, s=3, color='r')
            try:
                ax2.plot(data[Ys, Xs])
            except:
                print(Ys, Xs, "not in ", cols, rows) 
            ax2.set_xticks(np.linspace(0, len(Xs), 10))
            ax2.set_xticklabels(np.linspace(0,360, 10))
            ax2.set_ylim(0,255)
            plt.draw()
            plt.pause(0.01)

    plt.show()

    embed()



