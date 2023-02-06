import cv2
import pandas as pd
import numpy as np

from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import NoNorm
from matplotlib.colors import SymLogNorm
from matplotlib import cm

from test_dlis_load_function import dlis_loader

def main ():
    # file = '.\\data\\58-32_processed_image\\DLIS_XML_ProcessedImages\\University_of_Utah_MU-ESW1_FMI-HD_7390-7527ft_Run3.dlis'
    # #file = '.\\data\\58-32_FMI_DLIS_XML\\University_of_Utah_MU_ESW1_FMI_HD_2226_7550ft_Run1.dlis'
    # #file = '.\\data\\58-32_FMI_DLIS_XML\\University_of_Utah_MU_ESW1_FMI_HD_7440_7550ft_Run2.dlis'

    # curves_frame = dlis_loader(file)
    # fmi_dyn = curves_frame['FMI_DYN']
    # fmi_dyn[fmi_dyn==-9999]=np.nan
    # fmi_stat = curves_frame['FMI_STAT']
    # fmi_stat[fmi_stat==-9999]=np.nan

    # for k, fmi in {'stat':fmi_stat, 'dyn':fmi_dyn}.items():
    #     if k=='stat': norm = SymLogNorm(linthresh=0.1, linscale=1.0, vmin=np.nanmin(fmi) if np.nanmin(fmi)>0.0 else 0.0, vmax=np.nanmax(fmi), clip=False, base=10)
    #     else: norm = Normalize(vmin=np.nanmin(fmi) if np.nanmin(fmi)>0.0 else 0.0, vmax=np.nanmax(fmi), clip=False)
    #     mapper = cm.ScalarMappable(norm=norm, cmap=cm.YlOrBr)
    #     fmi_raster = mapper.to_rgba(fmi, bytes=True)
    #     cv2.imwrite(f'fmi_{k}.png', cv2.cvtColor(fmi_raster, cv2.COLOR_RGBA2BGRA))
    #     cv2.imshow("image", cv2.cvtColor(fmi_raster, cv2.COLOR_RGBA2BGRA))
    #     cv2.waitKey()
    
    img = cv2.imread('fmi_dyn.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    cv2.imshow("image", blur_gray)
    cv2.waitKey()

    low_threshold = 10
    high_threshold = 50
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    cv2.imshow("image", edges)
    cv2.waitKey()

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    
    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imwrite(f'lines_edges.png', lines_edges)
    cv2.imshow("image", lines_edges)
    cv2.waitKey()

if __name__ == "__main__":
    main()