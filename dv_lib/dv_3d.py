import numpy as np
import cv2 as cv
import concurrent.futures
from itertools import repeat


def preprocess(scan):
    z_index = scan.shape.index(
        [unique for unique in list(set(scan.shape))
         if scan.shape.count(unique) == 1][0]
    )
    scan = scan.T if z_index == 0 else scan
    scan_pp = scan.copy().T if z_index == 0 else scan.copy()

    scan_pp = np.where(scan_pp > scan.std(0) * 2, scan_pp, 0.0)
    scan_pp = np.where(scan_pp > scan.std(1) * 2, scan_pp, 0.0)
    # scan_pp = np.where(scan_pp.T > scan.T.std(0) * 2, scan_pp.T, 0.0).T
    # scan_pp = np.where(scan_pp > scan.std() * 1, 255, 0)
    scan_pp = np.array(scan_pp, dtype=np.uint8)

    for i in range(scan_pp.shape[2]):
        slice = scan_pp[:, :, i]
        # slice = cv.fastNlMeansDenoising(slice,h=20,templateWindowSize=10,searchWindowSize=21)
        # slice = cv.threshold(slice,80,255,cv.THRESH_BINARY)[1]
        scan_pp[:, :, i] = cv.ximgproc.thinning(slice)
    return np.array(scan_pp)


def preprocess_multiprocess(scan, chunksize):
    z_index = scan.shape.index(
        [unique for unique in list(set(scan.shape))
         if scan.shape.count(unique) == 1][0]
    )
    scan = scan.T if z_index == 0 else scan
    scan_pp = scan.copy().T if z_index == 0 else scan.copy()

    scan_pp = np.where(scan_pp > scan.std(0) * 2, scan_pp, 0.0)
    scan_pp = np.where(scan_pp > scan.std(1) * 2, scan_pp, 0.0)
    # scan_pp = np.where(scan_pp.T > scan.T.std(0) * 2, scan_pp.T, 0.0).T
    # scan_pp = np.where(scan_pp > scan.std() * 1, 255, 0)
    scan_pp = np.array(scan_pp, dtype=np.uint8)

    def preprocess_slice(slice):
        slice = cv.ximgproc.thinning(slice)
        # slice = cv.threshold(slice, 127, 255, cv.THRESH_BINARY)[1]
        return slice

    with concurrent.futures.ThreadPoolExecutor() as executor:
        scan_pp = [
            slice
            for slice in executor.map(preprocess_slice, scan_pp, chunksize=chunksize)
        ]
    return np.array(scan_pp)


def radial_normals(points, zdip=None, inward=False):
    """
    Compute radially distributed normals from the volume center over the xy plane
    where:
            points: is a numpy.array with shape (n,3)

            inwards (bol defatul:False): is a bolean that if True calculate the normals pointing inwards the volume center

            zdip (float defatul:None): if None computes the dip squerically from the volume center, otherwise takes the provided value

            chunksize (int defatul:1000): size of chunks for each paralel worker

            output: is a numpy.array with shape (n,3) containing the normals of each given point
    """
    i = -1 if inward else 1
    xmean = points[:, 0].mean()
    ymean = points[:, 1].mean()
    zmean = points[:, 2].mean()
    dist = points[:, 2].max() - points[:, 2].min()
    normals = []
    for x, y, z in points:
        azimuth = np.arctan2((x - xmean), (y - ymean))
        dip = np.arcsinh((z - zmean) / dist) if zdip == None else zdip
        normals.append([i * np.sin(azimuth), i * np.cos(azimuth), dip])
    return np.array(normals)


def normal(x, y, z, a, xmean, ymean, zmean, dist, zdip=None):
    azimuth = np.arctan2((x - xmean), (y - ymean))
    dip = np.arcsinh((z - zmean) / dist) if zdip == None else zdip
    return a * np.sin(azimuth), a * np.cos(azimuth), dip


def radial_normals_multiprocess(points, zdip=None, inward=False, chunksize=1000):
    """
    Compute radially distributed normals from the volume center over the xy plane
    where:
            points: is a numpy.array with shape (n,3)

            inwards (bol defatul:False): is a bolean that if True calculate the normals pointing inwards the volume center

            zdip (float defatul:None): if None computes the dip squerically from the volume center, otherwise takes the provided value

            chunksize (int defatul:1000): size of chunks for each paralel worker

            output: is a numpy.array with shape (n,3) containing the normals of each given point
    """
    a = -1 if inward else 1
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    xmean = points[:, 0].mean()
    ymean = points[:, 1].mean()
    zmean = points[:, 2].mean()
    dist = points[:, 2].max() - points[:, 2].min()

    # def normal(x, y, z, a, xmean, ymean, zmean, dist):
    #     azimuth = np.arctan2((x-xmean), (y-ymean))
    #     dip = np.arcsinh((z-zmean)/dist)
    #     return a*np.sin(azimuth), a*np.cos(azimuth), dip

    with concurrent.futures.ProcessPoolExecutor() as executor:
        normals = [
            normal
            for normal in executor.map(
                normal,
                x,
                y,
                z,
                repeat(a),
                repeat(xmean),
                repeat(ymean),
                repeat(zmean),
                repeat(dist),
                repeat(zdip),
                chunksize=chunksize,
            )
        ]
    return np.array(normals)
