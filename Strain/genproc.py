#---------some simple codes to facilitate calculations, filtering---------
import numpy as np
from math import log10,floor
import obspy
import scipy


def roundsigfigs(x,n):
    """
    :param         x: value to round
    :param         n: number of significant figures to keep
    """

    if isinstance(x,int) or isinstance(x,float):
        nr = int(floor(log10(abs(x))))
        x = round(x,n-nr-1)
    else:
        x = np.array([roundsigfigs(xi,n) for xi in x])

    return x


def closest(xvals,x):
    """
    :param      xvals:    set of sorted values
    :param          x:    values of interest
    :return        ix:    index of closest value
    """

    xvals = np.atleast_1d(xvals)
    x = np.atleast_1d(x)

    # index before and after
    ix = np.searchsorted(xvals,x,'left')

    # in range
    ix = np.maximum(ix,0)
    ix = np.minimum(ix,len(xvals)-2)

    
    # before or after?
    dx1 = np.abs(xvals[ix]-x)
    dx2 = np.abs(xvals[ix+1]-x)
    dx2 = dx2<dx1

    ix[dx2] = ix[dx2]+1
    ix = np.maximum(ix,0)

    return ix



def addfiltmask(st,msk):
    """
    :param        st: waveforms or trace
    :param       msk: a set of waveforms with masks
    """
    
    if isinstance(st,obspy.Trace):
        # the mask
        ms = msk.data.astype(bool)
        try:
            # if there's already a mask, combine them
            st.data.mask=np.logical_or(st.data.mask,ms)
        except:
            st.data=np.ma.masked_array(st.data,mask=ms)

    elif isinstance(st,obspy.Stream):

        for tr in st:
            # select mask
            ms = msk.select(id=tr.id)[0]

            # add filter
            addfiltmask(tr,ms)

    

def prepfiltmask(st,tmask=3.):
    """
    :param        st: waveforms or trace
    :param     tmask: time window to mask within some interval or endpoint, 
                         in seconds
    :return      msk: a set of waveforms with masks
    """

    if isinstance(st,obspy.Trace):

        # allowable values
        try:
            ms=np.logical_or(st.data.mask,np.isnan(st.data.data))
            data=st.data.data
        except:
            ms = np.isnan(st.data)
            data=st.data

        # times
        tm=np.arange(0.,data.size)

        # interpolate
        if np.sum(~ms):
            data[ms]=scipy.interp(tm[ms],tm[~ms],data[~ms])
        else:
            data[:]=0.
        
        # and copy data
        st.data = data    

        # to mask
        nwin = int(tmask/st.stats.delta)
        nwin = np.maximum(nwin,1)
        win = scipy.signal.boxcar(nwin*2+1)
        ms = ms.astype(float)
        ms = scipy.signal.convolve(ms,win,mode='same')

        # also the beginning and end
        if tmask != 0.:
            ms[0:nwin+1]=1.
            ms[-nwin:]=1.

        # place in trace
        ms = np.minimum(ms,1.)
        msk = st.copy()
        msk.data = ms

    elif isinstance(st,obspy.Stream):
        msk = obspy.Stream()

        for tr in st:
            mski = prepfiltmask(tr,tmask=tmask)
            msk.append(mski)

    return msk
