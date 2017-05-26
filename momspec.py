import obspy
import numpy as np
import tremorabhi
import numpy as fft
import spectrum
import seisproc
import matplotlib.pyplot as plt
import graphical
import general
import matplotlib
from matplotlib import gridspec
from matplotlib.patches import Polygon
from matplotlib.dates import date2num


def plotmom(typ='short',square=False):
    """
    :param    typ: which length of tremor moment to import
    """

    # read the tremor moment rate
    st = tremorabhi.readcat(typ=typ)

    # select the moment
    tri = st.select(channel='moment')[0].copy()

    # plot
    tlm = np.array([date2num(tri.stats.starttime),
                    date2num(tri.stats.endtime)])
    tlm = np.round(tlm)

    # reference timing
    trefp = date2num(tri.stats.starttime)
    tm = tri.times()/86400. + trefp

    plt.close()
    f = plt.figure(figsize=([10,6]))
    p=plt.axes([0.12,0.15,0.87,0.8])

    # normalize
    mn = np.mean(tri.data[~tri.data.mask])
    tri.data = tri.data / mn

    p.plot_date(tm,tri.data,linestyle='-',marker=None,
                color='navy',linewidth=1.5)
    
    p.set_xlim(tlm)

    p.set_ylim([0,np.max(tri.data)*1.05])
    p.set_yticks(np.arange(0,np.max(tri.data)*1.05))

    fs = 'xx-large'

    p.set_ylabel('tremor beam amplitude',fontsize=fs)

    p.set_xlim(tlm)
    p.tick_params(axis='x',labelsize=fs)
    p.tick_params(axis='y',labelsize=fs)

    tcks = np.arange(np.min(tm),np.max(tm)+.1,1.)
    p.set_xticks(tcks)
    import matplotlib.dates as mdates
    p.fmt_xdata =mdates.DateFormatter('%d-%b')
    p.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    f.autofmt_xdate()


    graphical.printfigure('VLtremamp',f)
    


def plotmomspec(fvl=None,freq=None,fvls=None):
    """
    :param   fvl: power spectra of moment
    :param  freq: frequencies in Hz
    :param  fvls: power spectra of moment squared
    """

    if fvl is None:
        fvl = fvls
        fvls = None
        lbl = 'squared'
    elif fvls is None:
        lbl = 'moment'
    else:
        lbl = 'both'
    
    plt.close()
    f = plt.figure()
    p = plt.axes()

    scl = 3600.
    xmn = scl/(1.*3600.)
    ii = np.logical_and(freq*scl>xmn*.7,freq*scl<xmn/.7)
    #mn = np.interp(xmn,freq*scl,fvl)
    mn = np.median(fvl[ii])

    hh,=p.plot(freq*scl,fvl,color='blue')
    if fvls is not None:
        h=[hh]
        hh,=p.plot(freq*scl,fvls,color='r',zorder=1)
        h.append(hh)
        lbls = ['beam amplitude','beam amplitude squared']
        lg=p.legend(h,lbls,loc='lower left',fontsize='small')


    p.set_xscale('log')
    p.set_yscale('log')

    p.set_xlabel('frequency (hour$^{-1}$)')
    p.set_ylabel('power (over mean squared)')


    xlm = np.array([2./84600,np.max(freq)])*scl
    nlbl=['1','4/3','2']
    n=[1,4./3.,2]
    for ni in range(0,len(n)):
        nn=n[ni]
        p.plot(xlm,np.power(xlm/xmn,-nn)*mn,color='dimgray',linestyle='-',zorder=1,
               linewidth=2)
        xvl = xlm[0]*1.5**(ni+1)
        yvl = np.power(xvl/xmn,-nn)*mn
        p.text(xvl,yvl,'n='+str(nlbl[ni]),horizontalalignment='center',
               verticalalignment='center',backgroundcolor='white')

    ii = np.logical_and(freq*scl>xlm[0],freq*scl<=xlm[1])
    ylm = np.exp(general.minmax(np.log(fvl[ii]),1.1))
    p.set_ylim(ylm)
    p.set_xlim(xlm)

    graphical.printfigure('VLtmomspec_'+lbl,f)
    

def tmomspec(typ='short',square=False):
    """
    :param    typ: which length of tremor moment to import
    :param square: look at moment rate squared instead
    :return  fvl: power spectra, normalized to sinusoid 
                   amplitudes as a fraction of the mean
    :return freq: frequencies in Hz
    """

    # read the tremor moment rate
    st = tremorabhi.readcat(typ=typ)

    # select the moment
    tri = st.select(channel='moment')[0].copy()

    # interpolate to fill gaps
    msk = seisproc.prepfiltmask(tri,tmask=0)
    
    # square?
    if square:
        tri.data = np.power(tri.data,2)
    
    # remove a mean
    tri.data = (tri.data - np.mean(tri.data)) / np.mean(tri.data)

    # choose number of tapers
    NW = 5

    # get the spectrum
    tpr,evl = spectrum.dpss(tri.data.size,NW=NW)
    fvl = np.multiply(tpr,tri.data.reshape([tri.data.size,1]))
    fvl = np.fft.rfft(fvl,n=tri.data.size,axis=0)
    fvl = np.mean(np.power(np.abs(fvl),2),axis=1)
    freq = np.fft.rfftfreq(tri.data.size,d=tri.stats.delta)
    #fvl = fvl / (tri.data.size * 0.5**0.5) * tri.data.size
    fvl = fvl / (tri.data.size**2. / 4) * tri.data.size

    return fvl,freq


def plotspec(fdata,ndata,freq,lbls,tlk,scalc):

    plt.close()
    f = plt.figure(figsize=(9,9))
    gs,p=gridspec.GridSpec(1,1),[]
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p  = np.array(p)
    pm = p.reshape([1,1])
    gs.update(left=0.1,right=0.96)
    gs.update(bottom=0.1,top=0.9)
    gs.update(hspace=0.05,wspace=0.05)


    Ns = fdata.shape[1]
    cols = graphical.colors(Ns)
    cols = graphical.colors(fdata.shape[2])

    # smooth?
    fav = 0
    dfreq = np.median(np.diff(freq))
    if fav > 0:
        ngwin = int(np.round(fav/dfreq))
        ngwin = ngwin + 1 - ngwin%2
        gwin = scipy.signal.gaussian(ngwin,fav/dfreq)
        gwin = gwin / np.sum(gwin)

        # timing
        freq=scipy.signal.convolve(freq,gwin,'valid')

        isp = ngwin/5
        iget = np.arange(0,freq.size,isp)
    else:
        iget = np.arange(0,freq.size)

    # scale frequency to hours
    fscl = 3600.
    freq = freq * fscl
    xmn = 1./(6*3600)*fscl
    xlm = np.array([0.5/(86400.)*fscl,np.max(freq)])


    h=[]
    for k in range(0,len(lbls)):
        ch = lbls[k].replace('-na','')
        mn = []
        tlbls = []
        vmn,vmx=float('inf'),-float('inf')
        for n in range(0,fdata.shape[2]):
            if scalc[k,n]:
                scl = np.interp(xmn,freq,fdata[:,k,n])
                scl = 1

                # noise
                for m in range(0,ndata.shape[2]):
                    vl=ndata[:,k,m,n]/scl
                    if fav>0:
                        vl=scipy.signal.convolve(vl,gwin,'valid')
                    p[0].plot(freq[iget],vl[iget],color=cols[n],
                              label=lbls[k],linestyle='--')
                    vmn = np.min(np.append(vl,vmn))
                    vmx = np.max(np.append(vl,vmx))
                    
                # data
                vl=fdata[:,k,n]/scl
                if fav>0:
                    vl=scipy.signal.convolve(vl,gwin,'valid')
                hh,=p[0].plot(freq[iget],vl[iget],color=cols[n],
                              label=lbls[k],linestyle='-',linewidth=1)
                vmn = np.min(np.append(vl,vmn))
                vmx = np.max(np.append(vl,vmx))

                lbli = tlk[ch][n][0].strftime('%d.%b.%Y')
                lbli = lbli + ' - ' + tlk[ch][n][1].strftime('%d.%b.%Y')
                tlbls.append(lbli)

                mn.append(np.interp(xmn,freq,vl))
                h.append(hh)
        mn = np.mean(mn)
        for nn in [2,3,4]:
            p[0].plot(xlm,np.power(xlm/xmn,-nn)*mn,color='dimgray',linestyle='-',zorder=1,
                      linewidth=2)
            xvl = xlm[0]*1.3**(nn-1)
            yvl = np.power(xvl/xmn,-nn)*mn
            p[0].text(xvl,yvl,'n='+str(nn),horizontalalignment='center',
                      verticalalignment='center',backgroundcolor='white')
        p[0].set_ylim([vmn/1.05,vmx*1.05])
        lg=p[0].legend(h,tlbls,loc='upper right',fontsize='small')

    p[0].set_xscale('log')
    p[0].set_yscale('log')
    p[0].set_xlabel('frequency (hours$^{-1}$)')
    if scl!=1.:
        p[0].set_ylabel('normalized power spectra')
    else:
        p[0].set_ylabel('power spectra')

    p[0].set_xlim(xlm)
    p[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    fs = 'medium'
    for ph in p:
        rax = ph.twiny()
        rax.set_xlim(np.divide(1.,np.flipud(xlm)))
        rax.set_xscale('log')
        rax.set_xlabel('period (hours)',fontsize=fs)
        rax.tick_params(axis='x',labelsize=fs)
        rax.invert_xaxis()
        #rax.set_yticks([500,200,100,50,20])
        rax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #rax.set_yscale('log')

    nm = 'VLplotspec'
    if scl!=1:
        nm = nm + '_normalized'
    graphical.printfigure(nm,f)
