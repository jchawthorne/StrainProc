import numpy as np
from math import log10,floor

def roundsigfigs(x,n):
    if isinstance(x,int) or isinstance(x,float):
        nr = int(floor(log10(abs(x))))
        x = round(x,n-nr-1)
    else:
        x = np.array([roundsigfigs(xi,n) for xi in x])

    return x

def masknans(x):
    """
    :param      x:  array or masked array
    :return     x:  masked array, with nans masked
    """

    if isinstance(x,np.ma.masked_array):
        x.mask = np.logical_or(x.mask,np.isnan(x))
    else:
        x = np.ma.masked_array(x,mask=np.isnan(x))

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

def minmax(x,bfr=1.):
    """
    :param      x:   set of values
    :param    bfr:   how much to multiply the limits by (default: 1.)
    :return   lms:   limits
    """

    # minmax
    lms = np.array([np.min(x),np.max(x)])

    lms = np.mean(lms)+np.diff(lms)*bfr*np.array([-.5,.5])

    return lms
