import numpy as np
from matplotlib.dates import date2num
import obspy

def timdata(tr):
    """
    :param     tr:   a waveform or trace
    :return   tim:   times
    :return  data:   data
    """
    
    if isinstance(tr,obspy.Trace):
        # all the times
        start = date2num(tr.stats.starttime)
        end = date2num(tr.stats.endtime)
        tim = np.linspace(start,end,tr.stats.npts)

        # data
        data = tr.data


    return tim,data

    
