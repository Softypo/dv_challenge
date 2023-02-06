import numpy as np

'''
Compute radially distributed normals from the volume center over the xy plane
where:
        points: is a numpy.array with shape (n,3)
        inwards (defatul:False): is a bolean that if True calculate the normals pointing inwards the volume center

        output: is a numpy.array with shape (n,3) containing the normals of each given point
'''


def radial_norms(points, inward=False):
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
