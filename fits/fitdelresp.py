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


def prepfit(st=None,fpar={}):
    """
    :param       st:  waveforms
    :param     fpar:  fit parameters
    :return     stf:  waveforms for the instantaneous response
    """

    # timing relative to input
    fpar['rsptspl']=fpar.get('rsptspl',None)
    if fpar['rsptspl'] is None:
        fpar['rsptmax']=fpar.get('tmax',30.)
        fpar['rsptmin']=fpar.get('tmin',st[0].stats.delta*3)
        fpar['rspN']=fpar.get('rspN',10)
        fpar['rsptspl']=np.linspace(np.log(fpar['rsptmin']),
                                    np.log(fpar['rsptmax']),
                                    fpar['rspN']-1)
        fpar['rsptspl']=np.exp(fpar['rsptspl'])
        fpar['rsptspl']=np.append(np.array([0.]),fpar['rsptspl'])
    else:
        fpar['rspN']=len(fpar['rsptspl'])
        fpar['rsptmax']=np.max(fpar['rsptspl'])
        fpar['rsptmin']=np.min(fpar['rsptspl'][fpar['rsptspl']>0.])
    
    # extract the one that looks like pressure
    stf = st.select(channel='WPR').copy()
    stf[0].stats.channel='IRP'

    return stf
    

def formod(st,fpar=None,X=None):
    """
    :param     st:  waveforms
    :param   fpar:  fit parameters
    :param      X:  current fit results
    :return     M:  column in forward model
    """

    # create a forward model
    M = np.ndarray([st[0].stats.npts,fpar['rspN']])
    M = np.atleast_2d(M)
    M = M.reshape([st[0].stats.npts,fpar['rspN']])

    # relevant times
    tms = st[0].times()/86400.

    # channel with input
    stf = st.select(channel='IRP')

    

    import code
    code.interact(local=locals())


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

    
