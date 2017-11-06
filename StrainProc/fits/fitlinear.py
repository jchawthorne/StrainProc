import numpy as np
import obspy
import os
from . import fitdefault


def updatedpar(*args,**kwargs):
    """
    to update the parameters
    nothing to do, so just pass to the defaults
    """

    result = fitdefault.updatedpar(*args,**kwargs)
    
    return result

# def calcdpar(*args,**kwargs):
#     """
#     to update the parameters
#     nothing to do, so just pass to the defaults
#     """

#     result = fitdefault.calcdpar(*args,**kwargs)
    
#     return result


def calcdpar(st,fpar=None,X=None):
    """
    :param     st:  waveforms
    :param   fpar:  fit parameters
    :param      X:  current fit results
    :return     M:  columns in forward model, 
                    per change in constant
    """

    # mean time from fit start and end times
    tmn = fpar['starttime']-st[0].stats.starttime
    tmn = tmn + (fpar['endtime']-fpar['starttime'])/2.

    # create a forward model, in strain per day
    M = st[0].times()
    M = (M - tmn)/86400.

    # reshape
    M = np.atleast_2d(M)
    M = M.reshape([st[0].stats.npts,1])

    return M


def updatepar(*args,**kwargs):
    """
    to update the parameters
    nothing to do, so just pass to the defaults
    """

    result = fitdefault.updatepar(*args,**kwargs)
    
    return result


def prepfit(st=None,fpar=None):
    """
    :param       st:  waveforms
    :param     fpar:  fit parameters
    :return  result:  as in fitdefault.prepfit
    """

    # nothing to do here, so just pass everything to the default
    result = fitdefault.prepfit(st=st,fpar=fpar)

    return result
    

def formod(st,fpar=None,X=None):
    """
    :param     st:  waveforms
    :param   fpar:  fit parameters
    :param      X:  current fit results
    :return     M:  column in forward model
    """

    # mean time from fit start and end times
    tmn = fpar['starttime']-st[0].stats.starttime
    tmn = tmn + (fpar['endtime']-fpar['starttime'])/2.

    # create a forward model, in strain per day
    M = st[0].times()
    M = (M - tmn)/86400.

    # reshape
    M = np.atleast_2d(M)
    M = M.reshape([st[0].stats.npts,1])

    return M


def pred(st,fpar=None,X=None,sta=None):
    """
    :param     st:  waveforms
    :param   fpar:  fit parameters---ignored
    :param      X:  fit results, including X['constant']
    :param    sta:  any other values necessary for the prediction
    :return     M:  column in forward model
    """

    # mean time from fit start and end times
    tmn = fpar['starttime']-st[0].stats.starttime
    tmn = tmn + (fpar['endtime']-fpar['starttime'])/2.

    # predicted values
    prd = st[0].times()
    prd = (prd - tmn)/86400.

    # to values
    prd = X['linear']*prd

    return prd
