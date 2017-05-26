import obspy
import numpy as np
# default functions for a given fit
 
def prepfit(st=None,fpar=None):
    """
    to create any waveforms to add to the data before processing
    :param    st:  waveforms
    :param  fpar:  parameters
    :return  stf:  waveforms again
    """

    # just return
    stf=obspy.Stream()

    return stf


def formod(st=None,fpar=None,X=None):
    """
    to create a forward model---a column or columns of M
    :param    st:  waveforms
    :param  fpar:  parameters
    :param     X:  current fit results
    :return    M:  an Nx0 array 
    """

    # create a forward model
    M = np.ndarray([st[0].stats.npts,0])
    M = np.atleast_2d(M)
    M = M.reshape([st[0].stats.npts,0])

    return M

def updatedpar(st=None,fpar=None,X=None,Xdiff=None):
    """
    to create a forward model for changes---a column or columns of M
    :param    st:  waveforms
    :param  fpar:  parameters
    :param     X:  current fit results
    :param Xdiff:  preferred differences to update
    """

    # nothing to do
    updt = False

    return updt

def calcdpar(st=None,fpar=None,X=None):
    """
    update any relevant parameters
    :param    st:  waveforms
    :param  fpar:  parameters
    :param     X:  current fit results
    :return    M:  an Nx0 array 
    """

    # create a forward model
    M = np.ndarray([st[0].stats.npts,0])
    M = np.atleast_2d(M)
    M = M.reshape([st[0].stats.npts,0])

    return M


def pred(st=None,fpar=None,X=None,sta=None):
    """
    to create a predicted contribution
    :param    st:  waveforms
    :param  fpar:  parameters
    :param     X:  the fit parameters
    :param    sta:  any other values necessary for the prediction
    :return  stf:  waveforms again
    """

    # create a forward model
    prd = np.zeros(st[0].stats.npts)

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

    # nothing to do
    dne = True

    return dne


