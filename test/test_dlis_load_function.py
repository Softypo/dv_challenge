import numpy as np
from dlisio import dlis


def dlis_loader(file):

    with dlis.load(file) as dlis_file:
        frames = {
            frame.name: [frame, frame.curves()] for logical_file in dlis_file for frame in logical_file.frames
        }
        # for logical_file in dlis_file:
        #     for frame in logical_file.frames:
        #         curves_frame = frame.curves()
        #     # for channel in f.channels:
        #     #    curves_channel = channel.curves()
    return frames


def main():
    file = '.\\data\\input_data\\Mckenzie_delta_FMI_SLB_CA\\mallik-5L38_dlis_original\\mallik-5L38_fmi_057pup.dlis'
    # file = '.\\data\\input_data\\geothermal_FMI_SLB_US\\DLIS_XML\\University_of_Utah_MU_ESW1_FMI_HD_7440_7550ft_Run2.dlis'

    frames = dlis_loader(file)
    TDEP = frames['1B'].channels[0]
    c = TDEP.curves()
    fmi_dyn = frames['FMI_DYN']
    fmi_dyn[fmi_dyn == -9999] = np.nan
    fmi_stat = frames['FMI_STAT']
    fmi_stat[fmi_stat == -9999] = np.nan


if __name__ == "__main__":
    main()
