from . import fitdefault
import os
import numpy as np
import obspy
import math
 
def updatedpar(*args,**kwargs):
    """
    to update the parameters
    nothing to do, so just pass to the defaults
    """

    result = fitdefault.updatedpar(*args,**kwargs)
    
    return result


def calcdpar(*args,**kwargs):
    """
    to update the parameters
    nothing to do, so just pass to the defaults
    """

    result = fitdefault.calcdpar(*args,**kwargs)
    
    return result

def updatepar(*args,**kwargs):
    """
    to update the parameters
    nothing to do, s just pass to the defaults
    """

    result = fitdefault.updatepar(*args,**kwargs)
    
    return result


def prepfit(st,fpar):
    """
    :param     st: waveforms
    :param   fpar: fit parameters
    """

    # extract and return the atmospheric pressure trace
    stf = st.select(channel='RDO').copy()

    return stf

def formod(st,fpar=None,X=None):
    """
    :param     st:  waveforms
    :param   fpar:  fit parameters
    :param      X:  current fit results
    :return     M:  column in forward model
    """
    
    # just grab the atmospheric pressure
    stf = st.select(channel='RDO')
    M = stf[0].data

    # create a forward model
    M = np.atleast_2d(M)
    M = M.reshape([st[0].stats.npts,1])

    return M

def pred(st,fpar=None,X=None,sta=None):
    """
    :param     st:  waveforms
    :param   fpar:  fit parameters
    :param      X:  current fit results
    :param    sta:  any other values necessary for the prediction
    :return   prd:  column in forward model
    """
    # just grab the atmospheric pressure
    stf = (st+sta).select(channel='RDO')
    stf = stf.merge()
    prd = stf[0].data.copy()

    # and multiply
    prd = prd * X['atm']

    return prd

