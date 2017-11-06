import numpy as np
import obspy
import os
from . import fitdefault

def prepfit(st=None,fpar=None):
    """
    :param       st:  waveforms
    :param     fpar:  fit parameters
    :return  result:  as in fitdefault.prepfit
    """

    # nothing to do here, so just pass everything to the default
    result = fitdefault.prepfit(st=st,fpar=fpar)

    # set a decay parameter
    fpar['expdec']=fpar.get('expdec',{})
    if not isinstance(fpar['expdec'],dict):
        fpar['expdec']=dict.fromkeys(fpar['chfit'],fpar['expdec'])
    fpar['expdeclast']=fpar.get('expdeclast',{})

    # allow variation in the decay parameters?
    fpar['expdeclim']=fpar.get('expdeclim',{})
    if (fpar['expdeclim'] is 0) or (fpar['expdeclim'] is 0.):
        ons= np.array([1.,1.]).reshape([1,2])
        fpar['expdeclim']={}
    else:
        ons=np.array([0.2,10.]).reshape([1,2])
        if not isinstance(fpar['expdeclim'],dict):
            fpar['expdeclim']=dict.fromkeys(fpar['chfit'],fpar['expdeclim'])

    for ch in fpar['chfit']:
        fpar['expdec'][ch]=fpar['expdec'].get(ch,300.)
        fpar['expdec'][ch]=np.atleast_1d(fpar['expdec'][ch]).astype(float)
        dlms=fpar['expdec'][ch].reshape([fpar['expdec'][ch].size,1])
        fpar['expdeclim'][ch]=fpar['expdeclim'].get(ch,dlms*ons)
        fpar['expdeclim'][ch]=np.atleast_2d(fpar['expdeclim'][ch]).astype(float)

    # start time
    fpar['exptref']=fpar.get('exptref',st[0].stats.starttime-100.*86400)

    return result
    

def formod(st,fpar=None,X=None):
    """
    :param     st:  waveforms
    :param   fpar:  fit parameters
    :param      X:  current fit results
    :return     M:  columns in forward model
    """

    # channel
    ch = fpar['chfit'][0]

    # number of decay constants
    N = len(fpar['expdec'][ch])
    
    # create a forward model
    M = np.ones([st[0].stats.npts,N])
    M = np.atleast_2d(M)

    # times
    tm = st[0].times()+(st[0].stats.starttime-fpar['exptref'])
    for k in range(0,N):
        M[:,k]=np.exp(-tm/(86400.*fpar['expdec'][ch][k]))

    return M


def pred(st,fpar=None,X=None,sta=None):
    """
    :param     st:  waveforms
    :param   fpar:  fit parameters---ignored
    :param      X:  fit results, including X['constant']
    :param    sta:  any other values necessary for the prediction
    :return     M:  column in forward model
    """

    # initialize prediction
    prd = np.zeros(st[0].stats.npts)

    # channel
    ch = st[0].stats.channel

    # number of decay constants
    N = len(fpar['expdec'][ch])

    # times
    tm = st[0].times()+(st[0].stats.starttime-fpar['exptref'])
    for k in range(0,N):
        prd=prd+X['exp'][k]*np.exp(-tm/(86400.*fpar['expdec'][ch][k]))

    return prd

    
def updatepar(st=None,fpar=None,X=None,Xb=None,sta=None):
    """
    to update the preferred fits
    :param    st:  waveforms
    :param  fpar:  parameters
    :param     X:  the fit results
    :param    Xb:  the bootstrapped fit results
    :param   sta:  any other values necessary for the prediction
    :return  dne:  whether the updates are done (here true)
    """

    # defaults to no updates
    dne = 1

    for ch in X.keys():
        # check if there are variable lengths to consider
        if np.sum(np.diff(fpar['expdeclim'][ch],axis=1)):
            dlast=fpar['expdeclast'].get(ch,float('inf'))
            dlast=fpar['expdec'][ch]-dlast
            dlast=np.abs(np.divide(dlast,fpar['expdec'][ch]))

            # and that there's an update
            if np.max(dlast)>0.005:
                dne = -1

    # copy the decay parameters to X
    # save the frequencies to the output
    for ch in X.keys():
        X[ch]['expdec']=fpar['expdec'][ch]
        Xb[ch]['expdec']=fpar['expdec'][ch]
        X[ch]['exptref']=fpar['exptref']
        Xb[ch]['exptref']=fpar['exptref']

    
    return dne

def calcdpar(st,fpar=None,X=None):
    """
    :param     st:  waveforms
    :param   fpar:  fit parameters
    :param      X:  current fit results
    :return     M:  columns in forward model, 
                    per log change in the decay timescales
    """

    # channel
    ch = fpar['chfit'][0]

    # number of decay constants
    N = len(fpar['expdec'][ch])
    
    # create a forward model
    M = np.ones([st[0].stats.npts,N])
    M = np.atleast_2d(M)

    # times
    tm = st[0].times()+(st[0].stats.starttime-fpar['exptref'])
    tm = tm/86400.
    for k in range(0,N):
        cf=X[ch]['exp'][k]
        M[:,k]=cf*np.multiply(np.exp(-tm/fpar['expdec'][ch][k]),
                              tm/fpar['expdec'][ch][k]**2)

    return M

def updatedpar(st=None,fpar=None,X=None,Xdiff=None):
    """
    to create a forward model for changes---a column or columns of M
    :param    st:  waveforms
    :param  fpar:  parameters
    :param     X:  current fit results
    :param Xdiff:  preferred differences to update
    """

    # channel
    ch = fpar['chfit'][0]

    # the preferred shifts
    shfs = Xdiff[ch]['exp']

    # add a portion---not all the way to hope for better stability
    shfs=shfs*0.5

    # and no more than a percentage
    ampscl=np.divide(fpar['expdec'][ch]*0.1,np.abs(shfs))
    ampscl=np.minimum(ampscl,1)
    shfs=np.multiply(ampscl,shfs)

    # save the last values
    fpar['expdeclast']=fpar.get('expdeclast',{})
    fpar['expdeclast'][ch]=fpar['expdec'][ch]

    # add a portion---not all the way to hope for better stability
    fpar['expdec'][ch]=fpar['expdec'][ch]+shfs*0.5

    # within limits
    fpar['expdec'][ch]=np.maximum(fpar['expdec'][ch],
                                  fpar['expdeclim'][ch][:,0])
    fpar['expdec'][ch]=np.minimum(fpar['expdec'][ch],
                                  fpar['expdeclim'][ch][:,1])

    # did update
    updt = True
    
    return updt


