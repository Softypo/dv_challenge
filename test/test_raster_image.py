import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colors import SymLogNorm
from matplotlib import cm
import cv2

from test_dlis_load_function import dlis_loader


def raster_image(image, mode=None, save=False):
    try:
        if mode == 'dyn':
            norm = Normalize(vmin=np.nanmin(image) if np.nanmin(
                image) > 0.0 else 0.0, vmax=np.nanmax(image), clip=False)
        elif mode == 'stat':
            norm = SymLogNorm(linthresh=0.1, linscale=1.0, vmin=np.nanmin(image) if np.nanmin(
                image) > 0.0 else 0.0, vmax=np.nanmax(image), clip=False, base=10)
        else:
            raise ValueError('select mode stat or dyn')
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.YlOrBr)
        img_raster = mapper.to_rgba(image, bytes=True,)
        if save == 'True':
            save_image_raster(img_raster, sufix=mode)
        return img_raster
    except Exception:
        print(Exception)


def save_image_raster(img_raster, sufix=None):
    try:
        if sufix == None:
            sufix = ''
        else:
            sufix = f'_{sufix}'
        cv2.imwrite('C:/Users/Softypo/OneDrive/Documentos/_Burrito/Burrito_suite/data/output_data/'+f'img{sufix}.png',
                    cv2.cvtColor(img_raster, cv2.COLOR_RGBA2BGRA))
    except Exception:
        print(Exception)


def load_image_raster(img_raster_name):
    try:
        return cv2.imread('.\\data\\output_data\\'+img_raster_name)
    except Exception:
        print(Exception)


def show_image_raster(img_raster):
    try:
        return cv2.imshow("image", img_raster)
    except Exception:
        print(Exception)


def main():
    file = '.\\data\\input_data\\geothermal_FMI_SLB_US\\DLIS_XML\\University_of_Utah_MU-ESW1_FMI-HD_7390-7527ft_Run3.dlis'
    #file = '.\\data\\58-32_FMI_DLIS_XML\\University_of_Utah_MU_ESW1_FMI_HD_2226_7550ft_Run1.dlis'
    #file = '.\\data\\58-32_FMI_DLIS_XML\\University_of_Utah_MU_ESW1_FMI_HD_7440_7550ft_Run2.dlis'

    curves_frame = dlis_loader(file)
    tdep = curves_frame['TDEP']
    P1NO_FBST_S = curves_frame['P1NO_FBST_S']
    RB_FBST_S = curves_frame['RB_FBST_S']
    HAZIM_S = curves_frame['HAZIM_S']
    DEVIM_S = curves_frame['DEVIM_S']
    C1_S = curves_frame['C1_S']
    C2_S = curves_frame['C2_S']
    BS = curves_frame['BS']
    fmi_dyn = curves_frame['FMI_DYN']
    fmi_dyn[fmi_dyn == -9999] = np.nan
    fmi_stat = curves_frame['FMI_STAT']
    fmi_stat[fmi_stat == -9999] = np.nan

    fmi_raster_stat = raster_image(fmi_dyn, mode='dyn', save='True')
    fmi_raster_stat2 = load_image_raster('img_stat.png')
    show_image_raster(fmi_raster_stat2)


if __name__ == "__main__":
    main()
