from . import fitdefault
import numpy as np
import obspy
import os


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
                    per constant
    """

    # create a forward model
    M = np.ones(st[0].stats.npts)
    M = np.atleast_2d(M)
    M = M.reshape([st[0].stats.npts,1])

    return M

def updatepar(*args,**kwargs):
    """
    to update the parameters
    nothing to do, s just pass to the defaults
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

    # create a forward model
    M = np.ones(st[0].stats.npts)
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

    # create a forward model
    prd = X['constant']*np.ones(st[0].stats.npts)
    
    return prd

    
