import numpy as np


def dv_importFromRaw(filepath, T=True):

    try:
        with open(filepath, 'rb') as f:
            RAW = f.read
            data = np.frombuffer(RAW(),
                                 dtype=np.uint8,
                                 offset=0).reshape(1280, 768, 768)
            if T:
                values = data.T
            else:
                values = data

            return values

    except IOError:
        print("Error while opening the file!")


def main():
    # raw_vol = dv_importFromRaw("")
    pass


if __name__ == "__main__":
    main()
