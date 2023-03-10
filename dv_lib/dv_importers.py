import numpy as np


def dv_importFromVolume(filepath, RawVol=True, T=True):

    try:
        dtype = np.uint16 if RawVol else np.uint8
        with open(filepath, 'rb') as f:
            RAW = f.read
            data = np.frombuffer(RAW(),
                                 dtype=dtype,
                                 offset=0).reshape(1280, 768, 768)
            if T:
                values = data.T
            else:
                values = data

            return values

    except IOError:
        print("Error while opening the file!")


def dv_importFromDVCaliper(filepath, dropdup=False, T=False):

    try:
        with open(filepath, 'rb') as f:
            RAW = f.read
            elems, slices = np.frombuffer(RAW(16), dtype=np.int0)
            data = np.frombuffer(RAW(),
                                 dtype=np.float32,
                                 offset=0).reshape(slices, elems+1)

            if dropdup:
                unique_keys, indices = np.unique(data[:, 0], return_index=True)
                data = data[np.sort(indices)]

            depths = data[:, 0]
            if T:
                values = data[:, 1:].T
            else:
                values = data[:, 1:]

            return depths, values

    except IOError:
        print("Error while opening the file!")
