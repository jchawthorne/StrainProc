import os
import numpy as np
import obspy
import math
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


def prepfit(st,fpar):
    """
    :param     st: waveforms
    :param   fpar: fit parameters
    """

    # set default parameters
    fpar['fitdaily']=int(fpar['fitdaily'])
    fpar['dailyvar']=fpar.get('dailyvar',0.)

    # default reference time---always set to midnight
    dtr=obspy.UTCDateTime(st[0].stats.starttime.date)
    
    # initialize set of traces
    stf = obspy.Stream()

    # reference trace
    trrf = st[0].copy()

    # times for tides
    # in days relative to reference
    tim=(trrf.times()+(trrf.stats.starttime-dtr))/86400.

    # collect all the frequencies
    freqs=np.arange(1,fpar['fitdaily']+1,1.)

    for k in range(0,len(freqs)):
        # cosine values
        trrf.data = np.cos(tim*(2*math.pi*freqs[k]))
        # cosine labels
        trrf.stats.channel='D'+str(k+1)+'C'
        # add as a new trace
        stf.append(trrf.copy())

        # sine values
        trrf.data = np.sin(tim*(2*math.pi*freqs[k]))
        # sine labels
        trrf.stats.channel='D'+str(k+1)+'S'
        # add as a new trace
        stf.append(trrf.copy())

    # need to identify interpolation intervals?
    if fpar['dailyvar']:
        # number of intervals
        tdf = fpar['endtime']-fpar['starttime']
        Nd = int(np.ceil(tdf/86400./fpar['dailyvar']))
        
        # points to consider
        tspl = np.linspace(0,tdf,Nd+1)
        tspl = [fpar['starttime']+ tspli for tspli in tspl]
        
        fpar['dailysplits']=np.array(tspl)
        
    # return modified traces
    return stf

def dvaramps(st,fpar):
    """
    :param    st:  waveforms
    :param  fpar:  fit parameters
    :return amps:  amplitudes for the relevant intervals
    """

    if fpar['dailyvar']:
        # initialize
        Nd = len(fpar['dailysplits'])
        amps = np.zeros([st[0].stats.npts,Nd],dtype=float)
        
        # buffered times--extrapolate before and after
        tspl=fpar['dailysplits']
        tspl=np.append(st[0].stats.starttime,tspl)
        tspl=np.append(tspl,st[0].stats.endtime)
        
        # times relative to start time
        tms = st[0].times()
        if isinstance(tms,np.ma.masked_array):
            tms = tms.data
        tspl=tspl-st[0].stats.starttime


        for k in range(0,Nd):
            # for each split
            t1,t2,t3=tspl[k],tspl[k+1],tspl[k+2]
            
            # before
            ix=np.logical_and(tms>t1,tms<=t2)
            if k>0:
                amps[ix,k]=(tms[ix]-t1)/(t2-t1)
            else:
                amps[ix,k]=1.

            # after
            ix=np.logical_and(tms>=t2,tms<t3)
            if k<Nd-1:
                amps[ix,k]=(t3-tms[ix])/(t3-t2)
            else:
                amps[ix,k]=1.

    else:
        # just ones
        Nd = 1
        amps = np.ones([st[0].stats.npts,Nd],dtype=float)

    return amps


def formod(st,fpar=None,X=None):
    """
    :param     st:  waveforms
    :param   fpar:  fit parameters
    :param      X:  current fit results
    :return     M:  column in forward model
    """

    # collect all the frequencies
    freqs=np.arange(1,fpar['fitdaily']+1,1.)

    # amplitudes for the specified time ranges
    amps = dvaramps(st,fpar)
    Nd=amps.shape[1]

    # initialize
    M = np.ndarray([st[0].stats.npts,len(freqs)*2*Nd],dtype=float)


    for m in range(0,Nd):
        # go through and grab frequencies
        for k in range(0,len(freqs)):
            # index
            ix = len(freqs)*Nd*k+len(freqs)*m
            # add cosine and sine for this frequency
            sth = st.select(channel='D'+str(k+1)+'C')
            M[:,ix] = np.multiply(sth[0].data,amps[:,m])
            sth = st.select(channel='D'+str(k+1)+'S')
            M[:,ix+1] = np.multiply(sth[0].data,amps[:,m])

    # reshape
    M = np.atleast_2d(M)
    M = M.reshape([st[0].stats.npts,len(freqs)*2*Nd])

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

    # collect all the frequencies
    freqs=np.arange(1,fpar['fitdaily']+1,1.)

    # amplitudes for the specified time ranges
    amps = dvaramps(st,fpar)
    Nd=amps.shape[1]

    # initialize the prediction
    prd = np.zeros(st[0].stats.npts)

    # frequencies first so we can remake data if necessary
    for k in range(0,len(freqs)):
        sthc = st.select(channel='D'+str(k+1)+'C')
        sths = st.select(channel='D'+str(k+1)+'S')

        if (not sthc) or (not sths):
            stf = prepfit(st,fpari)
            sthc = stf.select(channel='D'+str(k+1)+'C')
            sths = stf.select(channel='D'+str(k+1)+'S')

        # go through shifts
        for m in range(0,Nd):
            # index
            ix = len(freqs)*Nd*k+len(freqs)*m

            # add to set
            prd = prd + np.multiply(sthc[0].data,amps[:,m])*X['daily'][ix]
            prd = prd + np.multiply(sths[0].data,amps[:,m])*X['daily'][ix+1]

    return prd



def plotdaily(X,fpar,lbl):
    """
    :param    X: fit results
    :param fpar: fit parameters
    :param  lbl: label for printing
    """
    # number of components
    M = len(X)

    # grid
    f = plt.figure(figsize=(10,7))
    gs=gridspec.GridSpec(M,1)
    ax = []
    for k in range(0,M):
        ax.append(plt.subplot(gs[k]))
    
    chn = X.keys()
    for k in range(0,M):
        ch = chn[k]
        f.sca(ax[k])
        # fit values
        data = X[ch]['daily']
        # times
        tim = fpar['endtime']-fpar['starttime']
        tim = np.linspace(0,tim,len(data))/86400.
        tim = tim+date2num(fpar['starttime'])
        hc,=plt.plot_date(tim,np.real(data),color='red',
                          linestyle='-',label='4pm peak')
        hs,=plt.plot_date(tim,np.imag(data),color='blue',
                          linestyle='-',label='4am peak')
        ax[k].set_ylabel(ch)
        lm = np.median(np.abs(data))*np.array([-1,1])*3
        ax[k].set_ylim(lm)
        #plt.grid(b=True,which='major',color='k',linestyle='--')
        ax[k].xaxis.grid()
        
    f.sca(ax[0])
    lg = plt.legend(handles=[hc,hs],loc='lower left',fontsize='small')
    f.autofmt_xdate()

    if lbl:
        fname='plotdaily_'+lbl
        fname=os.path.join(os.environ['FIGURES'],fname+'.pdf')
        pp=PdfPages(fname)
        pp.savefig(f)
        pp.close()
        plt.clf()
        plt.close(f)
    else:
        plt.show()
