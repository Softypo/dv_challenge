import numpy as np


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

            # depths = np.reshape(data[:, 0], (slices, 1))
            depths = data[:, 0]
            if T:
                values = data[:, 1:].T
            else:
                values = data[:, 1:]

            return depths, values

    except IOError:
        print("Error while opening the file!")


def main():
    depths, data = dv_importFromDVCaliper(
        ".\data\\665\DVCaliper_Files\mianpass.radii.suppressed-both-45-pcnt.dvcaliper")
    pass


if __name__ == "__main__":
    main()
