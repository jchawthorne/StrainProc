import os
import numpy as np
import obspy
import math
import tides
from . import fitdefault


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

    # nothing to do if there's no deletion
    if fpar['delfreq']==0:
        dne = True
    else:
        # check channels
        chh = X.keys()
        dne = True

        for ch in chh:
            # grab indices
            Xi,Xbi=X[ch]['tides'],Xb[ch]['tides']

            # convert tidal values to complex numbers
            Nf=len(fpar['tfreq'][ch])
            ix=np.arange(0,Nf*2,2)
            Xi = Xi[ix]+1j*Xi[ix+1]
            Xbi = Xbi[ix,:]+1j*Xbi[ix+1,:]

            # difference
            df = Xi.reshape([Nf,1])-Xbi
            frc = np.mean(np.abs(df),axis=1)
            frc = np.divide(frc,np.abs(Xi))
            
            # acceptable frequencies
            iok = frc<fpar['delfreq']
            if sum(iok)<Nf:
                dne=False
                # just delete one at a time
                idel=np.argmax(frc)
                fpar['tfreq'][ch]=np.delete(fpar['tfreq'][ch],idel)

    # save the frequencies to the output
    for ch in X.keys():
        X[ch]['tfreq']=fpar['tfreq'][ch]
        Xb[ch]['tfreq']=fpar['tfreq'][ch]
            
    return dne


def prepfit(st,fpar):
    """
    :param     st: waveforms
    :param   fpar: fit parameters
    """

    # default reference time for tides
    dtr = fpar.get('dtr',None)
    if dtr is None:
        dtr = obspy.UTCDateTime('2000-01-01')
        fpar['dtr']=dtr

    # frequencies
    setfrequencies(st,fpar)

    # delete frequencies
    fpar['delfreq']=fpar.get('delfreq',0.)

    # initialize set of traces
    stf = obspy.Stream()

    # reference trace
    trrf = st[0].copy()

    # times for tides
    # in days relative to reference
    tim=(trrf.times()+(trrf.stats.starttime-dtr))/86400.

    # collect all the frequencies
    freqs=np.array([])
    for ch in fpar['chfit']:
        freqs=np.append(freqs,fpar['tfreq'][ch])

    for k in range(0,len(freqs)):
        # cosine values
        trrf.data = np.cos(tim*(2*math.pi*freqs[k]))
        # cosine labels
        trrf.stats.channel='T'+str(k)+'C'
        trrf.stats.location = "{:.9}".format(freqs[k])
        # add as a new trace
        stf.append(trrf.copy())

        # sine values
        trrf.data = np.sin(tim*(2*math.pi*freqs[k]))
        # sine labels
        trrf.stats.channel='T'+str(k)+'S'
        # add as a new trace
        stf.append(trrf.copy())

    # return modified traces
    return stf


def setfrequencies(st,fpar):
    """
    :param     st: waveforms
    :param   fpar: fit parameters
    """

    # defaults
    fpar['tfreq']=fpar.get('tfreq',None)

    # check if the frequencies are given
    tgiven = True
    for ch in fpar['chfit']:
        try:
            fpar['tfreq'][ch]=np.atleast_1d(fpar['tfreq'][ch])
            fpar['tfreq'][ch]=fpar['tfreq'][ch].astype(float)
        except:
            tgiven = False

    if tgiven:
        fpar['tidespec']='specified'
    else:
        fpar['tidespec']=fpar.get('tidespec','snr')
        if fpar['tidespec'] is 'specified':
            fpar['tidespec']='snr'

    if fpar['tidespec'] is '5big':
        # just 5 largest periods?

        freqs=largetides()
        fpar['tfreq']=dict.fromkeys(fpar['chfit'],freqs)
        fpar['tidepar']=None


    elif fpar['tidespec'] is 'snr':
        # based on signal to noise availability

        # a parameter for this tidal selection
        fpar['tidepar']=fpar.get('tidepar',0.5)

        # read the available tides
        tdvl = tides.readcte()

        # select base on signal to noise ratio
        fpar['tfreq']=pickfreq(st,tdvl,snr=fpar['tidepar'])

    # remove anything to close to daily
    # need to get rid of redundant fits
    dfm=86400./(fpar['endtime']-fpar['starttime'])/100
    for k in range(0,fpar['fitdaily']):
        for ch in fpar['chfit']:
            freqs=fpar['tfreq'][ch]
            fpar['tfreq'][ch]=freqs[np.abs(freqs-(k+1))>dfm]



def largetides():
    """
    :return  tfreq: frequencies of the 5 largest tides
    """

    tfreq=np.array([12.4206,12.6583,12,23.9345,25.8193])
    tfreq=np.divide(24.,tfreq)

    return tfreq


def pickfreq(st,tdvl,snr=0.5):
    """
    :param     st:     waveforms
    :param   tdvl:     tides available
    :param    snr:    required signal to noise level (default: 0.5)
    :return tfreq:  lists of the frequencies for each component
    """

    # default snr
    if snr is None:
        snr = 0.5
    
    # copy and taper
    stf = st.copy()
    stf = stf.split().detrend()
    stf=stf.taper(type='cosine',max_percentage=0.5,max_length=3.*86400.)

    # high-pass filter at long periods
    stf=stf.filter('highpass',freq=1./100./86400.,corners=1,zerophase=True)

    # merge to resample
    stf.merge(fill_value=0)
    
    # frequencies
    ii = tdvl['freqs']>0.5
    td = tdvl.copy()
    td['freqs']=td['freqs'][ii]
    td['amp']=td['amp'][ii]
    td['ampu']=td['ampu'][ii]
    td['dnum']=td['dnum'][ii,:]
    td['degs']=td['degs'][ii]

    ifr = np.round(td['freqs'])
    ifr,ibel=np.unique(ifr,return_inverse=True)

    # initialize frequencies
    tfreq={}

    for tr in stf:
        # for each trace
        tfreqi = np.array([])
        
        tim = tr.times()
        data = tr.data
        ii = np.isnan(data)
        jj = np.logical_not(ii)
        data[ii] = np.interp(tim[ii],tim[jj],data[jj])
        
        # relevant amplitudes 
        amp = np.fft.fft(data)
        freq = np.fft.fftfreq(len(tr.data),d=tr.stats.delta/86400.)
        ii = freq>0
        amp = abs(amp[ii])
        freq = freq[ii]
        
        for k in range(0,len(ifr)):
            # find the largest predicted amplitude
            ii = np.where(ibel==k)
            jj = np.argmax(abs(td['amp'][ii]))
            mxvl = abs(td['amp'][ii][jj])
            
            # and the relevant scaling factor for it
            scl = np.interp(td['freqs'][ii][jj],freq,amp)/mxvl
            
            # predicted amplitudes for the whole group
            aprd = abs(td['amp'][ii])*scl
            
            # noise levels
            flm = ifr[k]+np.array([-1,1])*0.2
            kk = np.logical_and(freq>=flm[0],freq<=flm[1])
            ns = np.median(amp[kk])
            
            # acceptable values
            kk = aprd>ns*snr
            
            # add to list
            tfreqi=np.append(tfreqi,td['freqs'][ii][kk])

        # all frequencies for this component
        tfreq[tr.stats.channel] = tfreqi

    # return frequencies
    return tfreq



def formod(st,fpar=None,X=None):
    """
    :param     st:  waveforms
    :param   fpar:  fit parameters
    :param      X:  current fit results
    :return     M:  column in forward model
    """

    # frequencies here (there should only be one channel)
    freqs=fpar['tfreq'][fpar['chfit'][0]]

    # initialize
    M = np.ndarray([st[0].stats.npts,len(freqs)*2],dtype=float)

    # go through and grab tides
    for k in range(0,len(freqs)):
        # add cosine and sine for this tidal frequency
        pst = "{:.9}".format(freqs[k])
        sth = st.select(channel='T*C',location=pst)
        M[:,k*2] = sth[0].data
        try:
            M[sth[0].data.mask,k*2] = float('nan')
        except:
            pass
        sth = st.select(channel='T*S',location=pst)
        M[:,k*2+1] = sth[0].data
        try:
            M[sth[0].data.mask,k*2+1] = float('nan')
        except:
            pass


    # create a forward model
    M = np.atleast_2d(M)
    M = M.reshape([st[0].stats.npts,len(freqs)*2])

    return M

def pred(st,fpar=None,X=None,sta=None):
    """
    :param     st:  waveforms
    :param   fpar:  fit parameters
    :param      X:  current fit results
    :param    sta:  any other values necessary for the prediction
    :return   prd:  column in forward model
    """

    # copy in case we need more frequencies
    fpari = fpar.copy()
    fpari['chfit']=[st[0].stats.channel]

    # frequencies here (there should only be one channel)
    freqs=fpar['tfreq'][fpari['chfit'][0]]

    # initialize
    # create a forward model
    prd = np.zeros(st[0].stats.npts)


    # go through and grab tides
    for k in range(0,len(freqs)):
        # add cosine and sine for this tidal frequency
        pst = "{:.9}".format(freqs[k])
        sthc = st.select(channel='T*C',location=pst)
        sths = st.select(channel='T*S',location=pst)

        if not sthc:
            # may not have the components any more,
            # so remake them
            fpari['tfreq']={fpari['chfit'][0]:np.array(freqs[k])}
            stf=prepfit(st,fpari)

            sthc = stf.select(channel='T*C',location=pst)
            sths = stf.select(channel='T*S',location=pst)

        # add to set
        prd = prd + sthc[0].data * X['tides'][k*2]
        prd = prd + sths[0].data * X['tides'][k*2+1]

    return prd


def pickfreq(st,tdvl,snr=None):
    """
    :param  st:     waveforms
    :param  tdvl:     tides available
    :param  snr:    required signal to noise level (default: 0.5)
    :return tfreq:  lists of the frequencies for each component
    """

    # default snr
    if snr is None:
        snr = 0.5
    
    # copy and taper
    stf = st.copy()
    stf = stf.split().detrend()
    stf=stf.taper(type='cosine',max_percentage=0.5,max_length=3.*86400.)

    # high-pass filter at long periods
    stf=stf.filter('highpass',freq=1./100./86400.,corners=1,zerophase=True)

    # merge to resample
    stf.merge(fill_value=0)
    
    # frequencies
    ii = tdvl['freqs']>0.5
    td = tdvl.copy()
    td['freqs']=td['freqs'][ii]
    td['amp']=td['amp'][ii]
    td['ampu']=td['ampu'][ii]
    td['dnum']=td['dnum'][ii,:]
    td['degs']=td['degs'][ii]

    ifr = np.round(td['freqs'])
    ifr,ibel=np.unique(ifr,return_inverse=True)

    # initialize frequencies
    tfreq={}

    for tr in stf:
        # for each trace
        tfreqi = np.array([])
        
        tim = tr.times()
        data = tr.data
        ii = np.isnan(data)
        jj = np.logical_not(ii)
        data[ii] = np.interp(tim[ii],tim[jj],data[jj])
        
        # relevant amplitudes 
        amp = np.fft.fft(data)
        freq = np.fft.fftfreq(len(tr.data),d=tr.stats.delta/86400.)
        ii = freq>0
        amp = abs(amp[ii])
        freq = freq[ii]
        
        for k in range(0,len(ifr)):
            # find the largest predicted amplitude
            ii = np.where(ibel==k)
            jj = np.argmax(abs(td['amp'][ii]))
            mxvl = abs(td['amp'][ii][jj])
            
            # and the relevant scaling factor for it
            scl = np.interp(td['freqs'][ii][jj],freq,amp)/mxvl
            
            # predicted amplitudes for the whole group
            aprd = abs(td['amp'][ii])*scl
            
            # noise levels
            flm = ifr[k]+np.array([-1,1])*0.2
            kk = np.logical_and(freq>=flm[0],freq<=flm[1])
            ns = np.median(amp[kk])
            
            # acceptable values
            kk = aprd>ns*snr
            
            # add to list
            tfreqi=np.append(tfreqi,td['freqs'][ii][kk])

        # all frequencies for this component
        tfreq[tr.stats.channel] = tfreqi

    # return frequencies
    return tfreq


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
