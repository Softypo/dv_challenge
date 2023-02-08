import numpy as np
import cv2 as cv
import concurrent.futures
from itertools import repeat

def preprocess (scan):
    scan = scan.copy()
    for i in range(scan.shape[2]):
        slice = scan[:,:,i]
        #slice = cv.fastNlMeansDenoising(slice,h=20,templateWindowSize=10,searchWindowSize=21)
        #slice = cv.threshold(slice,80,255,cv.THRESH_BINARY)[1]
        scan[:,:,i] = cv.ximgproc.thinning(slice)
    return scan

def preprocess (scan):
    scan = scan.copy()
    for i in range(scan.shape[2]):
        slice = scan[:,:,i]
        #slice = cv.fastNlMeansDenoising(slice,h=20,templateWindowSize=10,searchWindowSize=21)
        #slice = cv.threshold(slice,80,255,cv.THRESH_BINARY)[1]
        scan[:,:,i] = cv.ximgproc.thinning(slice)
    return scan

def radial_normals(points, inward=False):
    '''
    Compute radially distributed normals from the volume center over the xy plane
    where:
            points: is a numpy.array with shape (n,3)
            inwards (defatul:False): is a bolean that if True calculate the normals pointing inwards the volume center

            output: is a numpy.array with shape (n,3) containing the normals of each given point
    '''
    i = -1 if inward else 1
    xmean = points[:, 0].mean()
    ymean = points[:, 1].mean()
    zmean = points[:, 2].mean()
    dist = points[:, 2].max() - points[:, 2].min()
    normals = []
    for x, y, z in points:
        azimuth = np.arctan2((x-xmean), (y-ymean))
        dip = np.arcsinh((z-zmean)/dist)
        normals.append([i*np.sin(azimuth), i*np.cos(azimuth), dip])
    return np.array(normals)

def normal(x, y, z, a, xmean, ymean, zmean, dist):
    azimuth = np.arctan2((x-xmean), (y-ymean))
    dip = np.arcsinh((z-zmean)/dist)
    return a*np.sin(azimuth), a*np.cos(azimuth), dip

def radial_normals_multiprocess(points, inward=False, chunksize=1000):
    """
    Compute radially distributed normals from the volume center over the xy plane
    where:
            points: is a numpy.array with shape (n,3).
            inwards (defatul:False): is a bolean that if True calculate the normals pointing inwards the volume center.
            chunksize (defatul:1000): size of chunks for each paralel worker.
            output: is a numpy.array with shape (n,3) containing the normals of each given point.
    """
    a = -1 if inward else 1
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    xmean = points[:, 0].mean()
    ymean = points[:, 1].mean()
    zmean = points[:, 2].mean()
    dist = points[:, 2].max() - points[:, 2].min()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        normals = [normal for normal in executor.map(normal, *(x, y, z, repeat(a), repeat(xmean), repeat(ymean), repeat(zmean), repeat(dist)), chunksize=chunksize)]
    return np.array(normals)