import vlfes
import seisproc
import scipy
import obspy
import Strain
import matplotlib.pyplot as plt
import graphical
import os
import general
import matplotlib
from matplotlib import gridspec
from matplotlib.patches import Polygon
from matplotlib.dates import date2num
import numpy as np
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid.inset_locator import inset_axes


def plotenvstrain(stl,ste,smn,emn):
    """
    :return    stl: strain rates (per second)
    :return    ste: envelopes
    :return    smn: strain rate means removed
    :return    emn: envelope means removed
    """

    ids = np.unique([tr.get_id() for tr in stl])
    ide = np.unique([tr.get_id() for tr in ste])
    Ns = len(ids)
    Ne = len(ide)

    plt.close()
    f = plt.figure(figsize=(9,12))
    gs,p=gridspec.GridSpec(Ns,Ne),[]
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p  = np.array(p)
    pm = p.reshape([Ns,Ne])

    # define bins
    sbn = np.linspace(-1,3,25)
    ebn = np.linspace(-0,2,25)
    Nha = np.zeros([len(sbn)-1,len(ebn)-1,Ne,Ns])
    vlse = np.ndarray([Ne,Ns],dtype=list)
    vlss = np.ndarray([Ne,Ns],dtype=list)

    for ke in range(0,Ne):
        stei = ste.select(id=ide[ke])
        emni = emn[stei[0].get_id()]
        etmn = np.array(emni.keys())
        for tre in stei:
            ii=np.argmin(np.abs(etmn-tre.stats.starttime.timestamp))
            datae = np.power(tre.data/emni[etmn[ii]],1)+1
            for ks in range(0,Ns):
                stli = stl.select(id=ids[ks])
                smni = smn[stli[0].get_id()]
                stmn = np.array(smni.keys())
                vlse[ke,ks]=np.ma.masked_array([],dtype=float,mask=False)
                vlss[ke,ks]=np.ma.masked_array([],dtype=float,mask=False)
                for trs in stli:
                    jj=np.argmin(np.abs(stmn-trs.stats.starttime.timestamp))
                    datas = trs.data/smni[stmn[jj]]+1
                    #pm[ks,ke].plot(datae,datas,linestyle='none',
                    #               marker = 'o',color = 'k')

                    if not isinstance(datas,np.ma.masked_array):
                        datas = np.ma.masked_array(datas,mask=False)
                    if not isinstance(datae,np.ma.masked_array):
                        datae = np.ma.masked_array(datae,mask=False)
                    iok = np.logical_and(~datas.mask,~datae.mask)
                    
                    # create a histogram
                    Nh,x,y = np.histogram2d(datas[iok],datae[iok],
                                            bins=[sbn,ebn])
                    Nha[:,:,ke,ks]=Nha[:,:,ke,ks]+Nh

                    # add values to set
                    vlse[ke,ks]=np.append(vlse[ke,ks],datae[iok])
                    vlss[ke,ks]=np.append(vlss[ke,ks],datas[iok])

    mdns = np.ndarray([len(ebn)-1,Ne,Ns],dtype=float)
    mdns = np.ma.masked_array(mdns,mask=False)

    extent=[ebn[0],ebn[-1],sbn[0],sbn[-1]]
    for ke in range(0,Ne):
        for ks in range(0,Ns):
            hh=pm[ks,ke].imshow(Nha[:,:,ke,ks],aspect='auto',
                                extent=extent,interpolation='none',
                                origin='lower',cmap='Reds')

            for m in range(0,sbn.size-1):
                ii = np.logical_and(vlse[ks,ke]>=ebn[m],
                                    vlse[ks,ke]<ebn[m+1])
                ii = np.logical_and(ii,~vlse[ks,ke].mask)
                ii = np.logical_and(ii,~vlss[ks,ke].mask)
                if np.sum(ii)>10:
                    mdns[m,ke,ks]=np.median(vlss[ks,ke][ii])
                else:
                    mdns.mask[m,ke,ks] = True
                    mdns[m,ke,ks] = float('nan')
            
            x,y = graphical.baroutlinevals(ebn,mdns[:,ke,ks])
            pm[ks,ke].plot(x,y,color='k',linewidth=2)

    for ph in pm.flatten():
        ph.set_xlabel('tremor amplitude / mean')
        ph.set_ylabel('strain rate / mean')

def plotgoodstrain(sts,tlk=None,flm=[2.,24./2]):
    """
    :param     sts:   stations to consider
    :param     tlk:   intervals to plot
    :param     flm:   frequency limit to filter to
    """

    # default intervals to check
    if tlk is None:
        tlk,tmax=vlfes.intervals(tfrc=1.,tlk=None)

    # shorten the intervals
    tlk,tmax=vlfes.intervals(tfrc=0.25,tlk=tlk)

    # which time intervals
    tlm = np.array([])
    for k in range(0,len(sts)):
        tr = sts[k]
        ch = tr.stats.station+'.'+tr.stats.channel
        ch = ch.replace('-na','')
        tlm = np.append(tlm,tlk[ch][1])
    tlm = [np.min(tlm),np.max(tlm)]
    tdf = 86400*round((tlm[1]-tlm[0])/86400)+tr.stats.delta*3
    tlm[0] = obspy.UTCDateTime(tlm[0].year,tlm[0].month,tlm[0].day)
    tlm[1] = tlm[0]+tdf



    

    Ns = len(sts)

    plt.close()
    f = plt.figure(figsize=(12,9))
    gs,p=gridspec.GridSpec(Ns,1),[]
    for gsi in gs:
        p.append(plt.subplot(gsi))
    gs.update(left=0.2,right=0.96)
    p  = np.array(p)
    pm = p.reshape([Ns,1])

    fs = 'xx-large'


    # estimate strain rates in an interval centered 30 days beforehand
    tmn = tlm[0]+(tlm[1]-tlm[0])/2.-30*86400.
    tlen = 20*86400
    rslp = {}
    for tr in sts:
        t1,t2=tmn-tlen/2-tr.stats.starttime,tmn+tlen/2-tr.stats.starttime
        v1 = np.interp(t1,tr.times(),tr.data)
        v2 = np.interp(t2,tr.times(),tr.data)
        rslp[tr.get_id()] = (v2-v1)/tlen

    # change to strain rate
    sts = sts.copy()
    for tr in sts:
        tr.data = np.diff(tr.data)/tr.stats.delta*86400/1.e-9
        tr.stats.starttime=tr.stats.starttime+tr.stats.delta/2.

    # noise intervals
    tshf = -30*86400 + np.array([0.,1.,2.,3.,4.])*tmax*1.3
    cols = graphical.colors(len(tshf),lgt=False)
    lbls=[]

    # buffer
    flm = np.array(flm)
    if np.sum(flm ==np.array([0,float('inf')]))!=2:
        tbuf = 10./np.min(flm[flm>0])
    else:
        tbuf = 0.


    for k in range(0,len(sts)):
        tr = sts[k].copy()
        ylm = np.array([],dtype=float)

        # which time intervals
        ch = tr.stats.station+'.'+tr.stats.channel
        lbls.append(ch)
        ch = ch.replace('-na','')
        #tlm = tlk[ch]

        # copy and filter if desired
        if tbuf>0:
            trr = tr.copy().trim(starttime=tlm[0]-tbuf,endtime=tlm[1]+tbuf)
            msk = seisproc.prepfiltmask(trr,tmask=0.)

            # remove a mean
            smn = np.mean(trr.data)
            trr.data = trr.data - smn
            sgn = np.sign(smn)

            trr.filter('bandpass',freqmin=flm[0]/86400.,freqmax=flm[1]/86400.,
                       zerophase=True)

            # add mean back
            trr.data = trr.data + smn

            seisproc.addfiltmask(trr,msk)
        else:
            trr = tr.copy()

        # best data
        trr = trr.trim(starttime=tlm[0],endtime=tlm[1])
        md = np.median(trr.data)
        N=trr.stats.npts

        # timing
        tm = trr.times()/86400+date2num(trr.stats.starttime)        

        for n in range(0,len(tshf)):
            # copy and filter if desired
            if tbuf>0:
                tri = tr.copy().trim(starttime=tlm[0]-tbuf+tshf[n],
                                     endtime=tlm[1]+tbuf+tshf[n])
                msk = seisproc.prepfiltmask(tri,tmask=0.)

                # remove a mean
                mn = np.mean(tri.data)
                tri.data = tri.data - mn

                tri.filter('bandpass',freqmin=flm[0]/86400.,freqmax=flm[1]/86400.,
                           zerophase=True)

                # add mean back in
                #tri.data = tri.data + mn

                seisproc.addfiltmask(tri,msk)
            else:
                tri = tr.copy()
            
            tri=tri.trim(starttime=tlm[0]+tshf[n],
                         endtime=tlm[1]+tshf[n]+5*trr.stats.delta)
            mdi = np.median(tri.data[0:N])
            p[k].plot_date(tm,sgn*(trr.data+tri.data[0:N]-mdi),linestyle='-',marker=None,
                           color=cols[n],linewidth=2)
            ylm = general.minmax(np.append(ylm,sgn*(trr.data+tri.data[0:N]-mdi)))
            
        p[k].plot_date(tm,sgn*trr.data,linestyle='-',marker=None,color='gray',
                       linewidth=3)
        ylm = general.minmax(np.append(ylm,sgn*trr.data))

        p[k].set_ylim(general.minmax(ylm,1.15))

    for ph in p:
        ph.set_xlim(general.minmax(tm))
        ph.plot(general.minmax(tm),[0,0],color='k',linestyle='--',zorder=1)
        ph.tick_params(axis='x',labelsize=fs)
        ph.tick_params(axis='y',labelsize=fs)
    p[-1].set_ylabel('strain rate at B004.E-N-na (10$^{-9}$ day$^{-1}$)',fontsize=fs)

    tcks = np.arange(np.min(tm),np.max(tm),0.5)

    p[-1].set_xticks(tcks)
    import matplotlib.dates as mdates
    p[-1].fmt_xdata =mdates.DateFormatter('%d-%b-%y %H:%M')
    p[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%y %H:%M'))
    f.autofmt_xdate()

    lbli = tlm[0].strftime('%d-%b-%Y_')+tlm[1].strftime('%d-%b-%Y')

    graphical.printfigure('VLplotgoodstrain_'+lbli,f)



def plotspec(fdata,ndata,freq,lbls,tlk,scalc,slp=None,mrate=True):

    plt.close()
    f = plt.figure(figsize=(9,9))
    gs,p=gridspec.GridSpec(1,1),[]
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p  = np.array(p)
    pm = p.reshape([1,1])
    gs.update(left=0.15,right=0.98)
    gs.update(bottom=0.1,top=0.9)
    gs.update(hspace=0.05,wspace=0.05)


    Ns = fdata.shape[1]
    cols = graphical.colors(Ns)
    cols = graphical.colors(fdata.shape[2])
    fs = 'xx-large'


    if slp is None:
        slp = np.ones([Ns,fdata.shape[2]],dtype=float)
    else:
        slp = np.power(slp,2)

    # if we want to plot moment rate spectra instead of moment spectra
    if mrate:
        freq[0]=1.
        freqi = freq.reshape([freq.size,1,1])
        fdata = np.multiply(fdata,np.power(freqi,2))*(2.*np.pi)**2
        freqi = freq.reshape([freq.size,1,1,1])
        ndata = np.multiply(ndata,np.power(freqi,2))*(2.*np.pi)**2
        freq[0]=0.
        nplt = [1,2]
    else:
        # don't normalize otherwise
        slp = np.ones([Ns,fdata.shape[2]],dtype=float)
        nplt = [2,3,4]

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
    iplt = np.logical_and(freq>xlm[0],freq<xlm[1])

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
                    vl=vl/slp[k,m]
                    if fav>0:
                        vl=scipy.signal.convolve(vl,gwin,'valid')
                    p[0].plot(freq[iget],vl[iget],color=cols[n],
                              label=lbls[k],linestyle='--',
                              linewidth=2)
                    vmn = np.min(np.append(vl[iplt],vmn))
                    vmx = np.max(np.append(vl[iplt],vmx))
                    
                # data
                vl=fdata[:,k,n]/scl
                vl=vl/slp[k,m]
                if fav>0:
                    vl=scipy.signal.convolve(vl,gwin,'valid')
                hh,=p[0].plot(freq[iget],vl[iget],color=cols[n],
                              label=lbls[k],linestyle='-',linewidth=1.5)
                vmn = np.min(np.append(vl[iplt],vmn))
                vmx = np.max(np.append(vl[iplt],vmx))

                lbli = tlk[ch][n][0].strftime('%d.%b.%Y')
                lbli = lbli + ' - ' + tlk[ch][n][1].strftime('%d.%b.%Y')
                tlbls.append(lbli)

                mn.append(np.interp(xmn,freq,vl))
                h.append(hh)
        mn = np.mean(mn)
        for ni in range(0,len(nplt)):
            nn = nplt[ni]
            p[0].plot(xlm,np.power(xlm/xmn,-nn)*mn,color='dimgray',linestyle='-',zorder=1,
                      linewidth=2)
            xvl = xlm[0]*1.3**(ni+1)
            xvl = xlm[1]/1.6/1.9**(ni)
            yvl = np.power(xvl/xmn,-nn)*mn
            p[0].text(xvl,yvl,'n='+str(nn),horizontalalignment='center',
                      verticalalignment='center',backgroundcolor='white',
                      fontsize=fs)
        p[0].set_ylim([vmn/1.05,vmx*1.05])
        lg=p[0].legend(h,tlbls,loc='upper right',fontsize='medium')

    p[0].set_xscale('log')
    p[0].set_yscale('log')
    p[0].set_xlabel('frequency (hours$^{-1}$)',fontsize=fs)
    if scl!=1.:
        p[0].set_ylabel('normalized power spectra',fontsize=fs)
    elif slp[0,0]!=1:
        p[0].set_ylabel('strain rate amplitude / mean squared',fontsize=fs)
    elif mrate:
        p[0].set_ylabel('strain rate amplitude squared',fontsize=fs)
    else:
        p[0].set_ylabel('strain amplitude squared',fontsize=fs)


    p[0].set_xlim(xlm)
    p[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    p[0].tick_params(axis='x',labelsize=fs)
    p[0].tick_params(axis='y',labelsize=fs)

    for ph in p:
        rax = ph.twiny()
        rax.set_xlim(np.divide(1.,np.flipud(xlm))/24.)
        rax.set_xscale('log')
        rax.set_xlabel('period (days)',fontsize=fs)
        rax.tick_params(axis='x',labelsize=fs)
        rax.invert_xaxis()
        #rax.set_yticks([500,200,100,50,20])
        rax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #rax.set_yscale('log')

    nm = 'VLplotspec'
    if scl!=1:
        nm = nm + '_normalized'
    if mrate:
        nm = nm + '_rate'
    if slp[0,0]!=1:
        nm = nm + '_bymean'
        
    graphical.printfigure(nm,f)


def plotspecwtrem(fdata,ndata,freq,lbls,tlk,scalc,slp=None,mrate=True,fvl=None,tfreq=None,fvls=None):

    plt.close()
    f = plt.figure(figsize=(9,9))
    gs,p=gridspec.GridSpec(1,1),[]
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p  = np.array(p)
    pm = p.reshape([1,1])
    gs.update(left=0.13,right=0.96)
    gs.update(bottom=0.1,top=0.9)
    gs.update(hspace=0.05,wspace=0.05)


    Ns = fdata.shape[1]
    cols = graphical.colors(Ns)
    cols = graphical.colors(fdata.shape[2])

    if slp is None:
        slp = np.ones([Ns,fdata.shape[2]],dtype=float)
    else:
        slp = np.power(slp,2)

    # if we want to plot moment rate spectra instead of moment spectra
    if mrate:
        freq[0]=1.
        freqi = freq.reshape([freq.size,1,1])
        fdata = np.multiply(fdata,np.power(freqi,2))*(2.*np.pi)**2
        freqi = freq.reshape([freq.size,1,1,1])
        ndata = np.multiply(ndata,np.power(freqi,2))*(2.*np.pi)**2
        freq[0]=0.
        nplt = [1,4./3,2]
        nlbl = ['1','4/3','2']
    else:
        # don't normalize otherwise
        slp = np.ones([Ns,fdata.shape[2]],dtype=float)
        nplt = [2,3,4]
        nlbl = ['2','3','4']

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
    xlm = np.array([0.5/(86400.)*fscl,np.max(tfreq)*fscl])
    xlm = np.array([0.5/(86400.)*fscl,1./20*fscl])
    xlmp = np.array([1./(30*86400.)*fscl,xlm[1]])
    iplt = np.logical_and(freq>xlm[0],freq<xlm[1])
    sfmax = fscl/(2.*3600)
    iget = iget[np.logical_and(freq[iget]>xlm[0],freq[iget]<sfmax)]
    fs = 'xx-large'

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
                    vl=vl/slp[k,m]
                    if fav>0:
                        vl=scipy.signal.convolve(vl,gwin,'valid')
                    # p[0].plot(freq[iget],vl[iget],color=cols[n],
                    #           label=lbls[k],linestyle='--')
                    # vmn = np.min(np.append(vl[iplt],vmn))
                    # vmx = np.max(np.append(vl[iplt],vmx))
                    
                # data
                vl=fdata[:,k,n]/scl
                vl=vl/slp[k,m]
                if fav>0:
                    vl=scipy.signal.convolve(vl,gwin,'valid')
                hh,=p[0].plot(freq[iget],vl[iget],color=cols[n],
                              label=lbls[k],linestyle='-',linewidth=1)
                vmn = np.min(np.append(vl[iplt],vmn))
                vmx = np.max(np.append(vl[iplt],vmx))

                lbli = tlk[ch][n][0].strftime('%d.%b.%Y')
                lbli = lbli + ' - ' + tlk[ch][n][1].strftime('%d.%b.%Y')
                tlbls.append(lbli)

                mn.append(np.interp(xmn,freq,vl))
                h.append(hh)
        mn = np.mean(mn)
    hl = []
    for ni in range(0,len(nplt)):
        nn = nplt[ni]
        xlmph = np.array([xlmp[0]/20,xlmp[1]*20])
        hh,=p[0].plot(xlmph,np.power(xlmph/xmn,-nn)*mn,color='dimgray',linestyle='-',zorder=1,
                      linewidth=2)
        hl.append(hh)
        xvl = xlm[0]*1.3**(ni+1)
        xvl = xlm[1]/4/1.9**(ni+1)
        yvl = np.power(xvl/xmn,-nn)*mn
        hh=p[0].text(xvl,yvl,'n='+nlbl[ni],horizontalalignment='center',
                     verticalalignment='center',backgroundcolor='white',
                     fontsize=fs)
        hl.append(hh)
    p[0].set_ylim([vmn/1.05,vmx*1.05])
    lg=p[0].legend(h,tlbls,loc='upper right',fontsize='medium')


    p[0].set_xscale('log')
    p[0].set_yscale('log')
    p[0].set_xlabel('frequency (hours$^{-1}$)',fontsize=fs)
    if scl!=1.:
        p[0].set_ylabel('normalized power spectra',fontsize=fs)
    else:
        p[0].set_ylabel('inferred power / total moment squared',fontsize=fs)
        
    #p[0].plot(xlm,[1,1],color='k',linestyle='--',zorder=1)

    p[0].tick_params(axis='x',labelsize=fs)
    p[0].tick_params(axis='y',labelsize=fs)
    p[0].set_xlim(xlmp)
    #p[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    p[0].set_aspect('equal')

    for ph in p:
        rax = ph.twiny()
        rax.set_xlim(np.divide(1.,np.flipud(xlmp))/24)
        rax.set_xscale('log')
        rax.set_xlabel('period (days)',fontsize=fs)
        rax.tick_params(axis='x',labelsize=fs)
        rax.invert_xaxis()
        #rax.set_aspect('equal')
        #rax.set_yticks([500,200,100,50,20])
        #rax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #rax.set_yscale('log')

    tfmin = 1./(10*3600)
    ii = np.logical_and(tfreq>tfmin,tfreq<=xlm[1])

    # add the tremor spectrum
    ht,=p[0].plot(tfreq[ii]*fscl,fvl[ii],color='black',linewidth=2)
    vmn = np.min(np.append(vmn,fvl[ii]))
    vmx = np.max(np.append(vmx,fvl[ii]))

    # add VLFEs
    # moment of 1.6 times background for 3 minutes
    # moment as fraction of daily slow slip moment
    mom = 0.6*(30./180.)
    # 300 per 30-day period
    nperday = 300./30.
    # what fraction of the daily cycles include an event
    frc = nperday/(86400./60)
    # so total energy observed per day, in moment
    mmom  = (mom/2)**2*frc
    print(mmom)

    # moment per event, as a fraction of daily moment
    mom = 10**((3.4-5.5)*1.5)
    # times number of events per day
    nperday = 60
    mmom = mom**2 * nperday

    vmn = np.min(np.append(vmn,mmom*.7))

    p[0].plot(fscl/120.,mmom,linestyle='none',marker='o',
              markersize=90,alpha=0.25)
    p[0].text(fscl/120.,mmom*1.5,'VLFEs',fontsize=fs,
              horizontalalignment='center',verticalalignment='center')

    # RTMs
    # moment per event, as a fraction of daily moment
    mom = 10**((3.8-5.5)*1.5)
    # times number of events per day
    nperday = 30
    mmom = mom**2 * nperday
    pr = 60*20*2.

    vmn = np.min(np.append(vmn,mmom*.7))

    p[0].plot(fscl/pr,mmom,linestyle='none',marker='o',
              markersize=60,alpha=0.25)
    p[0].text(fscl/pr,mmom*2,'RTMs',fontsize=fs,
              horizontalalignment='center',verticalalignment='center')



    # add RTRs
    # moment of 2 times background for 3 minutes
    # moment as fraction of daily slow slip moment
    mom = 0.8
    # 2 per day
    nperday = 1.
    # what fraction of the daily cycles include an event
    frc = nperday/(24./6.)
    # so total energy observed per day, in moment
    mmom  = (mom/2)**2*frc
    print(mmom)


    p[0].plot(fscl/(3.*3600.)/2.,mmom,linestyle='none',marker='o',
              markersize=50,alpha=0.25)
    p[0].text(fscl/(3.*3600.)/2,mmom*2,'RTRs',fontsize=fs,
              horizontalalignment='center',verticalalignment='center')

    p[0].plot(fscl/(12.4*3600.)/2.,.25**2,linestyle='none',marker='o',
              markersize=30,alpha=0.5)
    p[0].text(fscl/(12.4*3600.)/2,.25**2/2.,'tidal',fontsize=fs,
              horizontalalignment='center',verticalalignment='center')
    
    
    vmx = np.maximum(vmx,1.)
    p[0].set_ylim([vmn/1.05,vmx*1.05])

    ylm = [vmn/1.05,vmx*1.05]
    mxdf = np.maximum(ylm[1]/ylm[0],xlmp[1]/xlmp[0])
    p[0].set_ylim([ylm[1]/mxdf,ylm[1]])
    p[0].set_xlim([xlmp[0],xlmp[0]*mxdf])

    nm = 'VLplotspecwtrem'
    if scl!=1:
        nm = nm + '_normalized'
    if mrate:
        nm = nm + '_rate'
    if slp[0,0]!=1:
        nm = nm + '_bymean'
        
    graphical.printfigure(nm,f)

    ht.remove()
    graphical.printfigure(nm+'_notrem',f)

    for hh in hl:
        hh.remove()
    for hh in h:
        hh.remove()
    lg.remove()
    
    graphical.printfigure(nm+'_justvlfe',f)


    

def plotsoffset(dsum,tims,stns,prate=False):
    """
    :param     dsum: sum of data, with dimensions 1: time
                        2: events, from all to the rest
                        3: stations/components
    :param     tims: times in seconds relative to reference 
    :param     stns: labels for stations or stream with the traces
    :param    prate: plot the rate instead of the value (default: False)
    """

    # time spacing
    dtim = np.median(np.diff(tims))

    # smooth?
    tav = 60.*3
    if tav > 0:
        ngwin = int(np.round(tav/dtim))
        ngwin = ngwin + 1 - ngwin%2
        gwin = scipy.signal.gaussian(ngwin,tav/dtim)
        gwin = gwin / np.sum(gwin)

        # timing
        tvl=scipy.signal.convolve(tims,gwin,'valid')
    else:
        tvl = tims.copy()

    # intermediate timing if plotting the rate
    if prate:
        tvl = (tvl[0:-1]+tvl[1:])/2.

    # add station average if appropriate
    if dsum.ndim==2:
        dsum = dsum.reshape([dsum.shape[0],dsum.shape[1],1])
        stns = ['average']

    Ns=dsum.shape[2]
    Ne=dsum.shape[1]
        
    f = plt.figure(figsize=(9,12))
    gs,p=gridspec.GridSpec(Ns,1),[]
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p  = np.array(p)
    pm = p.reshape([Ns,1])

    cols = graphical.colors(Ne)

    evlb = [str(k) for k in range(1,Ne)]
    evlb = ['average']+evlb
    
    for ks in range(0,Ns):
        for ke in range(1,Ne):
            vl=scipy.signal.convolve(dsum[:,ke,ks],gwin,'valid')

            if prate:
                vl = np.diff(vl)
            p[ks].plot(tvl/60,vl,color=cols[ke-1],
                       label=evlb[ke])

        vl=scipy.signal.convolve(dsum[:,0,ks],gwin,'valid')
        if prate:
            vl = np.diff(vl)
        p[ks].plot(tvl/60,vl,label='average',color='black',linewidth=1.5)
        p[ks].set_ylabel(stns[ks]+' strain')

    p[0].legend(fontsize='small')
    p[-1].set_xlabel('time (minutes)')
    graphical.delticklabels(pm)

def plotnumratios(iok,tms,grp,stl):
    """
    :param       iok:  indices of the values used
    :param       tms:  the times used
    :param       grp:  the templates for each event
    :param       stl:  the strain data
    """

    # initialize
    plt.close()
    f = plt.figure(figsize=(9,12))
    gs,p=gridspec.GridSpec(3,1),[]
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p  = np.array(p)
    pm = p.reshape([3,1])

    # timing in date2num format
    t1 = np.min(tms)-86400.
    t2 = np.max(tms)-86400.
    tm = (tms-t1)/86400+date2num(t1)
    tma = np.repeat(tm.reshape([tm.size,1]),iok.shape[1],axis=1)

    # time bins
    tlm = [date2num(t1),date2num(t2)]
    bns = np.arange(tlm[0],tlm[1],1.)

    # number of observations by all
    agrp = np.unique(grp)
    Ngrp = np.ndarray([len(bns)-1,len(agrp)],dtype=float)
    Nall,trash = np.histogram(tma[iok],bins=bns)

    # number of observations by event
    agrp,lgrp = np.unique(grp),[]
    Ngrp = np.ndarray([len(bns)-1,len(agrp)],dtype=float)
    for k in range(0,len(agrp)):
        ioka = np.repeat((grp==agrp[k]).reshape([grp.size,1]),iok.shape[1],axis=1)
        ioka = np.logical_and(ioka,iok)
        Ngrp[:,k],trash = np.histogram(tma[ioka],bins=bns)
        lgrp.append(str(agrp[k]))

    # number of observations by station
    Ncmp = np.ndarray([len(bns)-1,len(stl)],dtype=float)
    lstat = []
    for k in range(0,len(stl)):
        Ncmp[:,k],trash = np.histogram(tma[iok[:,k],k],bins=bns)
        lstat.append(stl[k].stats.station+'.'+stl[k].stats.channel)

    x,y=graphical.baroutlinevals(bns,Nall,wzeros=True)
    hall = p[0].plot_date(x,y,color='k',
                          linestyle='-',marker=None)
    p[0].set_ylim([0,np.max(y)*1.1])

    cols = graphical.colors(len(agrp))
    mx=0.
    for k in range(0,len(agrp)):
        x,y=graphical.baroutlinevals(bns,Ngrp[:,k],wzeros=True)
        p[1].plot_date(x,y,label=lgrp[k],color=cols[k],
                       linestyle='-',marker=None)
        mx=np.max(np.append([mx],y))
    p[1].set_ylim([0,mx*1.1])
    lgg=p[1].legend(loc='upper left',fontsize='small')

    cols = graphical.colors(len(stl))
    mx=0.
    for k in range(0,len(stl)):
        x,y=graphical.baroutlinevals(bns,Ncmp[:,k],wzeros=True)
        p[2].plot_date(x,y,label=lstat[k],color=cols[k],
                       linestyle='-',marker=None)
        mx=np.max(np.append([mx],y))
    p[2].set_ylim([0,mx*1.1])
    lgs=p[2].legend(loc='upper left',fontsize='small')

    for ph in p:
        ph.set_xlim(tlm)
        ph.set_ylabel('number of ratios')

    import matplotlib.dates as mdates
    p[-1].fmt_xdata =mdates.DateFormatter('%d-%b')
    p[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    f.autofmt_xdate()
    
    graphical.printfigure('VLplotnumratios',f)


def plotrates(lrates,grp,stl,tinfo,tlong=0.5):

    # events to consider
    kys = tinfo.keys()
    N = len(tinfo)

    # initialize
    plt.close()
    f = plt.figure(figsize=(10,6.5))
    gs,p=gridspec.GridSpec(N,1),[]
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p  = np.array(p)
    pm = p.reshape([N,1])

    bins = 30
    cols = graphical.colors(len(stl))
    for k in range(0,len(kys)):
        kk = kys[k]
        # just events for this template
        iuse, = np.where(grp==kk)
        for m in range(0,len(stl)):
            N,bns = np.histogram(lrates[iuse,m],bins=bins)
            x,y=graphical.baroutlinevals(bns,N,wzeros=False)
            p[k].plot(x,y,color=cols[m])


def plotspredmap(stl):
    """
    :param     stl: waveforms with observations of interest
    """

    ylm = np.array([47.3, 49.4])
    xlm = np.array([-126.,-122.4])

    # stations and channels
    stn = np.unique([tr.stats.station for tr in stl])
    cmps = np.unique([tr.stats.channel for tr in stl])
    Ns,Nc=stn.size,cmps.size

    # initialize
    plt.close()
    f = plt.figure(figsize=(8,12))
    gs,p=gridspec.GridSpec(Ns,Nc),[]
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p  = np.array(p)
    pm = p.reshape([Ns,Nc])
    gs.update(left=0.1,right=0.9)
    gs.update(bottom=0.1,top=0.9)
    gs.update(hspace=0.05,wspace=0.05)


    # points for calculation
    x=np.linspace(xlm[0],xlm[1],100)
    y=np.linspace(ylm[0],ylm[1],100)
    xm,ym=np.meshgrid(x,y)

    cmps = [cmi.replace('-na','') for cmi in cmps]

    # moment
    mom = 10**(1.5*3.5+16.1-7)

    spred = np.ndarray([y.size,x.size,Ns,Nc],dtype=float)
    for ks in range(0,Ns):
        locobs = stl.select(station=stn[ks])[0]
        oloc = [locobs.stats.sac.stlo,locobs.stats.sac.stla]
        for kx in range(0,x.size):
            for ky in range(0,x.size):
                sloc = [x[kx],y[ky]]
                # calculate
                strike=(360.-(90-54.))
                strain=Strain.predict.calcstrain(locslip_deg=sloc,
                                                 locobs_deg=oloc,moment=mom,
                                                 strike=strike,dip=30,
                                                 rake=90,depth=40.e3,cmps=cmps)

                # save
                spred[ky,kx,ks,:]=strain[0]
              

    # graphical.delticklabels(pm)

    xsp = general.roundsigfigs(np.diff(xlm)/4,1)
    lontk = np.arange(round(xlm[0],1),round(xlm[1],1),xsp)

    ysp = general.roundsigfigs(np.diff(ylm)/4,1)
    lattk = np.arange(round(ylm[0],1),round(ylm[1],1),ysp)

    # basemap
    m = Basemap(llcrnrlon=xlm[0],llcrnrlat=ylm[0],
                urcrnrlon=xlm[1],urcrnrlat=ylm[1],
                projection='lcc',resolution='i',
                lat_0=ylm.mean(),lon_0=xlm.mean(),
                suppress_ticks=True)

    clm=np.max(np.abs([np.min(spred),np.max(spred)]))*np.array([-1.,1])

    xv=general.minmax(xlm,0.95)[0]
    yv=general.minmax(ylm,0.9)[1]
    xv,yv=m(xv,yv)

    # event locations
    ixt,loc,tinfo=vlfes.temploc(yr=2011)
    xl,yl=m(loc[:,0],loc[:,1])

    extent = np.append(xlm,ylm)
    for ks in range(0,Ns):
        for kc in range(0,Nc):
            h=m.imshow(spred[:,:,ks,kc],ax=pm[ks,kc],vmin=clm[0],vmax=clm[1],
                       cmap='bwr')
            m.drawcoastlines(ax=pm[ks,kc])
            lbl =[kc==0,kc==Nc-1,ks==0,ks==Ns-1]
            m.drawmeridians(lontk,labels=lbl,ax=pm[ks,kc])
            m.drawparallels(lattk,labels=lbl,ax=pm[ks,kc])
            ht=pm[ks,kc].text(xv,yv,stn[ks]+'.'+cmps[kc],
                              verticalalignment='top',
                              backgroundcolor='white')
            pm[ks,kc].plot(xl,yl,color='k',marker='*',linestyle='none')
    pc=plt.axes([0.2,0.05,0.6,0.03])
    cb=plt.colorbar(h,cax=pc,orientation='horizontal')
    cb.set_label('predicted strain per M3.5')

    graphical.printfigure('VLplotspredmap',f)
    


def plotmap():

    # initialize
    plt.close()
    f = plt.figure(figsize=(10,6.5))
    #gs=gridspec.GridSpec(1,1)
    #p = plt.subplot(gs[0])
    p  = plt.axes()
    
    # limits
    ylm = np.array([47.3, 49.4])
    xlm = np.array([-126.,-122.4])

    ylm2 = np.array([41, 55.])
    xlm2 = np.array([-131.,-111.])

    # basemap
    m = Basemap(llcrnrlon=xlm[0],llcrnrlat=ylm[0],
                urcrnrlon=xlm[1],urcrnrlat=ylm[1],
                projection='lcc',resolution='i',
                lat_0=ylm.mean(),lon_0=xlm.mean(),
                suppress_ticks=True)
    m.drawlsmask(land_color='whitesmoke',ocean_color='aliceblue',
                 lakes=True,
                 resolution='i',grid=1.25)
    m.drawcoastlines()
    xsp = general.roundsigfigs(np.diff(xlm)/4,1)
    lontk = np.arange(round(xlm[0],1),round(xlm[1],1),xsp)
    m.drawmeridians(lontk,labels=[1,0,0,1])

    ysp = general.roundsigfigs(np.diff(ylm)/4,1)
    lattk = np.arange(round(ylm[0],1),round(ylm[1],1),ysp)
    m.drawparallels(lattk,labels=[0,1,1,0])

    ps = p.get_position()
    p2 = plt.axes([.03,0.2,0.3,0.35])
    f.sca(p2)

    # basemap
    m2 = Basemap(llcrnrlon=xlm2[0],llcrnrlat=ylm2[0],
                 urcrnrlon=xlm2[1],urcrnrlat=ylm2[1],
                 projection='lcc',resolution='l',
                 lat_0=ylm2.mean(),lon_0=xlm2.mean(),
                 suppress_ticks=True)
    m2.drawlsmask(land_color='whitesmoke',ocean_color='aliceblue',
                  lakes=True,
                  resolution='f',grid=1.25)
    m2.drawcoastlines()
    m2.drawstates()
    m2.drawcountries()

    xsp = general.roundsigfigs(np.diff(xlm2)/4,1)
    lontk = np.arange(round(xlm2[0],1),round(xlm2[1],1),xsp)
    m2.drawmeridians(lontk,labels=[0,0,0,0])

    ysp = general.roundsigfigs(np.diff(ylm2)/4,1)
    lattk = np.arange(round(ylm2[0],1),round(ylm2[1],1),ysp)
    m2.drawparallels(lattk,labels=[0,0,0,0])


    bxx = xlm[np.array([0,0,1,1,0])]
    bxy = ylm[np.array([0,1,1,0,0])]
    bxx,bxy =  m2(bxx,bxy)
    plt.plot(bxx,bxy,color='blue',linewidth=2)

    ps = p.get_position()
    f.sca(p)

    # for vl in ['30','40','50']:
    #     fl = os.path.join(os.environ['DATA'],'TREMOR','CONTOURS',
    #                       'cont-'+vl)
    #     vls = np.loadtxt(fl)
    #     x,y = m(vls[:,0],vls[:,1])
    #     plt.plot(x,y,color='k')

    cfile=os.path.join(os.environ['DATA'],'SLAB','Cascadia','contours')
    cvls = np.loadtxt(cfile,dtype='float')
    for vl in [-30,-40,-50]:
        ii = cvls[:,2]==vl
        x,y=m(cvls[ii,1],cvls[ii,0])
        plt.plot(x,y,color='darkgray',linewidth=1)

    dr = os.path.join(os.environ['DATA'],'SLOWSLIP','Cascadia','Polygons')
    #vls = np.loadtxt(os.path.join(dr,'allsliptremor'))
    vls = np.loadtxt(os.path.join(dr,'aug11_tremor4'))
    x,y=m(vls[:,0]-360,vls[:,1])
    plt.plot(x,y,color='darkblue')

    # template locations for 2011
    ixt,loc,trash=vlfes.temploc(yr=2011)

    # colors for the VLFE times
    agrp = np.unique(ixt)
    coli = graphical.colors(len(agrp),lgt=True)
    icol = general.closest(agrp,ixt)

    x,y=m(-124.3,48.9)
    plt.text(x,y,'Vancouver Island',verticalalignment='center',
             fontsize=14,horizontalalignment='center',
             backgroundcolor='whitesmoke')
    x,y=m(-123.75,47.6)
    plt.text(x,y,'Olympic Peninsula',verticalalignment='center',
             fontsize=14,horizontalalignment='center',
             backgroundcolor='whitesmoke')


    x,y = m(loc[:,0],loc[:,1])
    for k in range(0,len(x)):
        plt.plot(x[k],y[k],linestyle='none',marker='*',markersize=20,
                 color=coli[icol[k]])
    
    

    # template locations for 2014
    #ixt,loc=vlfes.temploc(yr=2014)

    # x,y = m(loc[:,0],loc[:,1])
    # plt.plot(x,y,linestyle='none',marker='*',markersize=20,
    #          color='blue')

    # get stations and locations
    stns = Strain.readwrite.centcascstat()
    stns = np.append(vlfes.stattouse(2011),vlfes.stattouse(2014))
    stns = np.unique(stns)
    stns = vlfes.stattouse(2011)
    sloc = np.zeros([len(stns),2])
    for k in range(0,len(stns)):
        stdata=Strain.readwrite.pbometadata(stns[k])
        sloc[k,0]=stdata['LONG']
        sloc[k,1]=stdata['LAT']
    x,y = m(sloc[:,0],sloc[:,1])
    plt.plot(x,y,linestyle='none',marker='^',markersize=10,
             color='black')
    yshfs={'B007':-0.11,'B010':-0.07,'B009':0.01,'B006':-.16,
           'B004':-.08,'B003':-0.12,'B005':-.01}
    for sti in yshfs.keys():
        ix,=np.where(stns==sti)
        sloc[ix,1]=sloc[ix,1]+yshfs[sti]
    xshfs={'B003':-.1}
    for sti in xshfs.keys():
        ix,=np.where(stns==sti)
        sloc[ix,0]=sloc[ix,0]+xshfs[sti]

    x,y = m(sloc[:,0]+0.1,sloc[:,1])
    for k in range(0,len(stns)):
        plt.text(x[k],y[k],stns[k],fontsize='large')
    

    graphical.printfigure('VLFEmap',f)

def plotstrainshort(yr=2011,stns=None,sts=None,stl=None):
    """
    plot strain time series for the whole event
    """

    # stations
    if stns is None:
        stns=np.array(['B004','B003','B005','B007'])

    # get VLFE times
    tms,grp,xc = vlfes.readtimes(yr=yr)

    # central time
    tcent = obspy.UTCDateTime(2011,8,21,4,9,3)
    tget = np.array([tcent-60.*60,tcent+60*60])
    tplt = np.array([tcent-60.*10,tcent+60*10])

    fs = 'x-large'

    sts = sts.copy().trim(starttime=tget[0],endtime=tget[1])
    msk = seisproc.prepfiltmask(sts)
    sts = sts.copy().filter('lowpass',freq=1./30.,zerophase=True)
    #sts = stf

    # event central time
    tmids=vlfes.sstimes()
    ix=np.argmin(np.abs(tmids-tms[0]))
    tmid=tmids[ix]

    # times for plotting
    tvl=(tms-tmid)/86400.+date2num(tmid)

    # data
    if stl is None:
        stl = vlfes.readdata(stns,['na'],pfx='R')
        sts = vlfes.readdata(stns,['na'],pfx='L',tlm=tget)

    # remove a trend from beforehand
    t1,t2=tmid-86400*25,tmid-86400*15
    for tri in sts:
        tr=stl.select(station=tri.stats.station,channel=tri.stats.channel)[0]
        tlm=np.array([t1-tr.stats.starttime,t2-tr.stats.starttime])
        rt=np.interp(tlm,tr.times(),tr.data)
        rt=np.diff(rt)[0]/np.diff(tlm)[0]
        tri.data=tri.data-rt*tri.times()
        tref = tcent-tri.stats.starttime
        rt=np.interp(tref,tri.times(),tri.data)
        tri.data=tri.data-rt

    # time limit
    tlm=[date2num(tplt[0]),date2num(tplt[1])]
    tlm=(tplt-tmid)/86400.+date2num(tmid)

    # to plot
    N = len(stns)
    plt.close()
    f = plt.figure(figsize=(10,8.5))
    gs,p=gridspec.GridSpec(N,1,width_ratios=[1]),[]
    gs.update(left=0.1,right=0.97)
    gs.update(bottom=0.12,top=0.97)
    gs.update(hspace=0.12,wspace=0.1)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p).reshape([N,1])

    cmps=np.unique([tr.stats.channel for tr in stl])
    cols=graphical.colors(len(cmps))

    # colors for the VLFE times
    agrp = np.unique(grp)
    coli = graphical.colors(len(agrp),lgt=True)
    icol = general.closest(agrp,grp)
    
    for k in range(0,N):
        # for m in range(0,len(tvl)):
        #     tm = tvl[m]
        #     p[k,0].plot([tm,tm],[-1,1],linestyle='-',
        #                 color=coli[icol[m]])

        # for each station
        sti=sts.select(station=stns[k])
        ylm=np.array([],dtype=float)
        h=[]
        for m in range(0,len(cmps)):
            tr=sti.select(channel=cmps[m])[0]
            tr.data = tr.data * 1.e9
            tri=tr.copy().trim(starttime=tplt[0],endtime=tplt[1])
            ylm=general.minmax(np.append(ylm,tri.data))
            print(ylm)
            
            # times
            tim=(tr.times()+(tr.stats.starttime-tmid))/86400.\
                +date2num(tmid)
            f.sca(p[k,0])
            hh,=plt.plot_date(tim,tr.data,color=cols[m],
                              linestyle='-',marker=None,
                              linewidth=2)
            h.append(hh)

        p[k,0].set_ylim(general.minmax(ylm,1.1))
        p[k,0].set_xlim(tlm)
        p[k,0].set_ylabel('strain at '+stns[k])
        p[k,0].set_ylabel(stns[k]+' (ns)',fontsize=fs)
        #p[k,0].set_ylabel(stns[k],fontsize=fs)

    p[0,0].legend(h,cmps,loc='upper right',fontsize=fs)
    #h=np.array(h).reshape([2,N])

    graphical.delticklabels(p)
    #    p[-1,0].xaxis.set_major_locator
    import matplotlib.dates as mdates
    p[-1,0].fmt_xdata =mdates.DateFormatter('%H:%M')
    p[-1,0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    f.autofmt_xdate()

    for ph in p.flatten():
        ph.xaxis.set_tick_params(labelsize=fs)
        ph.yaxis.set_tick_params(labelsize=fs)
        ph.locator_params(nticks=3,axis='y')
        ph.set_yticks([ph.get_yticks()[1],ph.get_yticks()[-2]])


    
    nm = 'VLFEstrainshort_'+str(yr)
    graphical.printfigure(nm,f)



def plotstrainlong(yr=2011,stns=None,stl=None):
    """
    plot strain time series for the whole event
    """

    # stations
    if stns is None:
        stns=np.array(['B004','B003','B005','B007'])

    # get VLFE times
    tms,grp,xc = vlfes.readtimes(yr=yr)
    tms,grp = tms[xc>4.5],grp[xc>4.5]

    # event central time
    tmids=vlfes.sstimes()
    ix=np.argmin(np.abs(tmids-tms[0]))
    tmid=tmids[ix]

    # times for plotting
    tvl=(tms-tmid)/86400.+date2num(tmid)

    # data
    if stl is None:
        stl = vlfes.readdata(stns,['na-cv'],pfx='R')

    # remove a trend from beforehand
    t1,t2=tmid-86400*25,tmid-86400*15
    for tr in stl:
        tlm=np.array([t1-tr.stats.starttime,t2-tr.stats.starttime])
        rt=np.interp(tlm,tr.times(),tr.data)
        rt=np.diff(rt)[0]/np.diff(tlm)[0]
        tr.data=tr.data-rt*tr.times()
        dref=np.interp(tlm[1],tr.times(),tr.data)
        tr.data=tr.data-dref
        

    # time limit
    t1,t2=tmid-86400*30,tmid+86400*20
    tlm=[date2num(t1),date2num(t2)]

    # to plot
    N = len(stns)
    plt.close()
    f = plt.figure(figsize=(10,9.5))
    gs,p=gridspec.GridSpec(N+1,1,width_ratios=[1]),[]
    gs.update(left=0.12,right=0.97)
    gs.update(bottom=0.12,top=0.97)
    gs.update(hspace=0.12,wspace=0.1)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    phist=p[0]
    p=np.array(p[1:]).reshape([N,1])
    
    fs = 'x-large'
    cmps=np.unique([tr.stats.channel for tr in stl])
    cols=graphical.colors(len(cmps))


    # colors for the VLFE times
    agrp = np.unique(grp)
    coli = graphical.colors(len(agrp),lgt=True)
    icol = general.closest(agrp,grp)
    
    colh = graphical.colors(len(agrp),lgt=False)
    tbns = np.arange(tlm[0],tlm[1],1.)
    mx = 0.
    h=[]
    for k in range(0,len(agrp)):
        ii = grp==agrp[k]
        nper,bns = np.histogram(tvl[ii],bins=tbns)
        x,y=graphical.baroutlinevals(bns,nper,wzeros=False)
        mx = np.max(np.append(y,mx))
        hh,=phist.plot(x,y,color=colh[k],label=str(agrp[k]))
        h.append(hh)
    phist.set_xlim(tlm)
    phist.set_ylabel('# VLFEs',fontsize=fs)
    phist.set_ylim([0,mx*1.1])
    #phist.legend(loc='upper right',fontsize='small')


    for k in range(0,N):
        for m in range(0,len(tvl)):
            tm = tvl[m]
            p[k,0].plot([tm,tm],[-1000,1000],linestyle='-',
                        color=coli[icol[m]],linewidth=0.4)

        # for each station
        sti=stl.select(station=stns[k])
        ylm=np.array([],dtype=float)
        h=[]
        for m in range(0,len(cmps)):
            tr=sti.select(channel=cmps[m])[0]
            tri=tr.copy().trim(starttime=t1-3600,endtime=t2+3600)
            ylm=general.minmax(np.append(ylm,tri.data*1.e9))
            
            # times
            tim=(tr.times()+(tr.stats.starttime-tmid))/86400.\
                +date2num(tmid)
            f.sca(p[k,0])
            hh,=plt.plot_date(tim,tr.data*1.e9,color=cols[m],
                              linestyle='-',marker=None,
                              linewidth=2)
            h.append(hh)

        p[k,0].set_ylim(general.minmax(ylm,1.1))
        p[k,0].set_xlim(tlm)
        p[k,0].set_ylabel('strain\nat '+stns[k],fontsize=fs)
        p[k,0].set_ylabel(stns[k]+' (ns)',fontsize=fs)
        #p[k,0].set_yticks([-40,0,)
        p[k,0].locator_params(nticks=3,axis='y')

    p[1,0].legend(h,cmps,loc='upper right',fontsize=fs)
    #h=np.array(h).reshape([2,N])
    import matplotlib.dates as mdates
    p[-1,0].fmt_xdata =mdates.DateFormatter('%d-%b')
    p[-1,0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    f.autofmt_xdate()
    graphical.delticklabels(np.append([phist],p.flatten()).reshape([N+1,1]))

    phist.xaxis.set_tick_params(labelsize=fs)
    phist.yaxis.set_tick_params(labelsize=fs)
    phist.set_yticks([0,phist.get_yticks()[-1]])
    for ph in p.flatten():
        ph.xaxis.set_tick_params(labelsize=fs)
        ph.yaxis.set_tick_params(labelsize=fs)
        ph.locator_params(nticks=3,axis='y')
        ph.set_yticks([ph.get_yticks()[1],ph.get_yticks()[-2]])


    nm = 'VLFEstrainlong_'+str(yr)
    graphical.printfigure(nm,f)
        
    #import code
    #code.interact(local=locals())


def plotratehist(srate,lrate,srates,lrates,iok,
                 tshort=None,tlong=None,fnm=None):
    """
    :param      srate:  short-term rates
    :param      lrate:  long-term rates
    :param     srates:  short-term shifted rates
    :param     lrates:  long-term shifted rates
    :param        iok:  acceptable times
    :param     tshort:  length of vlfe rate (for labelling)
    :param      tlong:  length of long-term rate (for labelling)
    :param        fnm:  file name for printing
    """

    rts = np.divide(srate,lrate)
    rtss = np.divide(srates,lrates)
    
    # get median rates
    mds=[]
    for k in range(0,rtss.shape[1]):
        vl=rtss[:,k,:]
        mds.append(np.median(vl[iok]))
    mds=np.array(mds)

    # for comparison
    shfs=vlfes.pickshifts()
    k=np.argsort(mds)[len(mds)/2]
    #k=rtss.shape[1]/4
    rtsi=rtss[:,k,:]
    shfi=shfs[k]
    
    shflb = graphical.timelabel(np.abs(shfi),wdash=False,nsig=1)
    if shfi>0:
        shflb=shflb[0]+'\nafter\n'
    elif shfi<0:
        shflb=shflb[0]+'\nbefore\n'

    bns1=np.linspace(-10,20,20)
    bns2=np.linspace(-1,2.,20)
    
    f = plt.figure(figsize=(10,5))
    gs,p=gridspec.GridSpec(1,2,width_ratios=[1,1]),[]
    gs.update(left=0.1,right=0.97)
    gs.update(bottom=0.1,top=0.97)
    gs.update(hspace=0.1,wspace=0.2)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p)
    pm=p.reshape([1,2])

    cols=graphical.colors(2)
    h=[]

    # histogram for shifted vlfe times
    npers,trash=np.histogram(rtsi[iok],bins=bns1)
    x,y=graphical.baroutlinevals(bns1,npers,wzeros=True)
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('firebrick')
    ply.set_color('pink')
    ply.set_linewidth(3)
    ply.set_alpha(0.5)
    p[0].add_patch(ply)

    # histogram for vlfe times
    nper,trash=np.histogram(rts[iok],bins=bns1)
    x,y=graphical.baroutlinevals(bns1,nper,wzeros=True)
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('darkblue')
    ply.set_linewidth(3)
    ply.set_color('lightblue')
    ply.set_alpha(0.5)
    p[0].add_patch(ply)

    nmax1=np.maximum(np.max(nper),np.max(npers))*1.25
    p[0].set_ylim([0,nmax1])
    p[0].set_xlim(general.minmax(bns1))

    md = np.median(rts[iok])
    mdi = np.median(rtsi[iok])
    hs,=p[0].plot([mdi,mdi],[-1,nmax1*3],color='firebrick',linewidth=2)
    hv,=p[0].plot([md,md],[-1,nmax1*3],color='darkblue',linewidth=2)
    
    # histogram for shifted vlfe times
    npers,trash=np.histogram(mds,bins=bns2)
    x,y=graphical.baroutlinevals(bns2,npers,wzeros=True)
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('firebrick')
    ply.set_color('pink')
    ply.set_alpha(0.5)
    ply.set_linewidth(3)
    p[1].add_patch(ply)

    nmax=np.max(npers)*1.1
    p[1].set_ylim([0,nmax])
    p[1].set_xlim(general.minmax(bns2))

    p[1].plot([md,md],[-1,nmax*3],color='darkblue',linewidth=2)    

    if tshort is not None:
        ls = graphical.timelabel(tshort*86400,2,True)[0]+' '
    else:
        ls = 'short-term '

    if tlong is not None:
        ll = graphical.timelabel(tlong*86400,2,True)[0]+' '
    else:
        ll = 'long-term '
    xlab = ls+ 'strain rate / '+ll+'strain rate'

    fs = 'medium'
    fs = 13
    p[0].set_xlabel(xlab,fontsize=fs)
    p[1].set_xlabel(xlab,fontsize=fs)


    p[0].set_ylabel('number of VLFE observations',fontsize=fs)
    p[1].set_ylabel('number of median observations',fontsize=fs)

    for ph in p:
        ph.xaxis.set_tick_params(labelsize=fs)
        ph.yaxis.set_tick_params(labelsize=fs)

    lg=p[0].legend([hv,hs],['at VLFE times',shflb+' VLFE times'],
                   loc='center right',fontsize='medium')
    
    print('VLFE median: '+str(md))
    
    tf=f.transFigure.inverted()
    c1=tf.transform(p[0].transData.transform([md,nmax1*0.95]))
    c2=tf.transform(p[1].transData.transform([md,nmax*0.95]))
    arw={'arrowstyle':'->','color':'darkblue'}
    plt.annotate(' ',xy=c2,xytext=c1,xycoords='figure fraction',arrowprops=arw)

    c1=tf.transform(p[0].transData.transform([mdi,nmax1*0.9]))
    c2=tf.transform(p[1].transData.transform([mdi,nmax*0.3]))
    arw={'arrowstyle':'->','color':'firebrick'}
    plt.annotate(' ',xy=c2,xytext=c1,xycoords='figure fraction',arrowprops=arw)


    cmn = c1*.35+c2*.65
    cmn2 = cmn+np.array([.01,.01])
    txt = 'for a range of\nrandom time shifts'
    plt.annotate(txt,xy=cmn,xytext=cmn2,xycoords='figure fraction',
                 color='firebrick')

    if fnm:
        nm = 'VLFEratehist_'+fnm
        graphical.printfigure(nm,f)
        plt.close()

def plotmedwtime(rmds,vlms,tshorts,tlongs,fnm=None,rmdsi=[],vlmsi=[]):
    """
    :param    rmds:  median ratios
    :param    vlms:  limiting ratios
    :param tshorts:  short-term durations
    :param  tlongs:  long-term durations
    :param     fnm:  file name for printing
    """
    
    plt.close()
    f = plt.figure(figsize=(10,4.5))
    gs,p=gridspec.GridSpec(1,2,width_ratios=[1,1]),[]
    gs.update(left=0.08,right=0.93)
    gs.update(bottom=0.11,top=0.97)
    gs.update(hspace=0.1,wspace=0.23)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p)
    pm=p.reshape([1,2])

    # moment per day
    mwday = 5.6
    momday = 3.1623e+17

    xsl=np.arange(0,len(tlongs)).astype(float)/len(tlongs)
    xsl=(xsl-np.mean(xsl))*0.5

    # a reference rate
    xref = np.array([0.1,500])
    rt = 0.5
    vlref = rt*xref/1440.*momday
    p[1].plot(xref,vlref,linestyle='--',color='k')

    xvl,yvl=np.array([0.7,1]),np.array([3.3,4.1])
    yvl = np.power(10,(yvl-mwday)*1.5)*momday
    ix,iy=np.array([0,1,1,0,0]),np.array([0,0,1,1,0])
    xvl,yvl=xvl[ix],yvl[iy]

    ply = Polygon(np.vstack([xvl,yvl]).transpose())
    ply.set_color('lightgray')
    ply.set_alpha(0.5)
    p[1].add_patch(ply)
    
    xx=(xvl[0]*xvl[1])**0.5
    yy=(yvl[0]**0.25)*(yvl[2]**0.75)

    p[1].text(xx,yy,'seismic\nestimates')

    if not rmdsi:

        c1=np.array([60,60*rt/1440.*momday])
        c2=np.array([60*.65,60*rt/1440.*momday])
        arw={'arrowstyle':'->','color':'black'}
        p[1].annotate('constant moment rate',xy=c1,
                      xytext=c2,xycoords='data',
                      arrowprops=arw,horizontalalignment='right',
                      verticalalignment='center')
        
        rt = 0.1
        c1=np.array([50,60*rt/1440.*momday])
        c2=np.array([0.7*60,0.5*60*rt/1440.*momday])
        arw={'arrowstyle':'->','color':'black'}
        txt = 'decreasing but still\nhigher-than-average\nmoment rate\nfarther from VLFEs?'
        p[1].annotate(txt,xy=c1,xytext=c2,xycoords='data',
                      arrowprops=arw,horizontalalignment='center',
                      verticalalignment='top')
        
    cols = graphical.colors(len(tlongs))
    colsi = graphical.colors(len(tlongs)+len(rmdsi),lgt=True)
    colsi = colsi[len(tlongs):]

    ylm=np.array([],dtype=float)
    for k in range(0,len(tlongs)):
        spl = xsl[k]
        hshf = []
        for m in range(0,len(rmdsi)):
            yerr=[1-vlmsi[m][0,:,k],vlmsi[m][1,:,k]-1.]
            p[0].errorbar(tshorts*1440+spl,rmdsi[m][:,k],
                          yerr=yerr,color=colsi[m])
            hh,=p[0].plot(tshorts*1440+spl,rmdsi[m][:,k],
                          marker='o',linestyle='none',
                          markersize=10,color=colsi[m])
            hshf.append(hh)

            vl = np.multiply((rmdsi[m][:,k]-1.)*momday,tshorts)

            vl1=1.-vlmsi[m][0,:,k] 
            vl1=np.multiply((vl1-0.)*momday,tshorts)
            vl2=vlmsi[m][1,:,k]-1.
            vl2=np.multiply((vl2-0.)*momday,tshorts)
            
            ylm = np.append(ylm,general.minmax(vl))
            ylm = np.append(ylm,general.minmax(np.abs(vl-vl1)))
            ylm = np.append(ylm,general.minmax(np.abs(vl+vl2)))
            p[1].errorbar(tshorts*1440+spl,vl,yerr=[vl1,vl2],
                          color=colsi[m])
            p[1].plot(tshorts*1440+spl,vl,marker='o',linestyle='none',
                      markersize=10,color=colsi[m])

        yerr=[1-vlms[0,:,k],vlms[1,:,k]-1.]
        p[0].errorbar(tshorts*1440+spl,rmds[:,k],yerr=yerr,color=cols[k])
        hh,=p[0].plot(tshorts*1440+spl,rmds[:,k],
                      marker='o',linestyle='none',
                     markersize=10,color=cols[k])
        hh=[hh]+hshf

        vl = np.multiply((rmds[:,k]-1.)*momday,tshorts)

        vl1=1.-vlms[0,:,k] #+rmds[:,k]
        vl1=np.multiply((vl1-0.)*momday,tshorts)
        vl2=vlms[1,:,k]-1. #+rmds[:,k]
        vl2=np.multiply((vl2-0.)*momday,tshorts)
        
        ylm = np.append(ylm,general.minmax(vl))
        ylm = np.append(ylm,general.minmax(np.abs(vl-vl1)))
        ylm = np.append(ylm,general.minmax(np.abs(vl+vl2)))
        p[1].errorbar(tshorts*1440+spl,vl,yerr=[vl1,vl2],
                      color=cols[k])
        p[1].plot(tshorts*1440+spl,vl,marker='o',linestyle='none',
                  markersize=10,color=cols[k])
        #p[0].set_ylim([0.9,2])


    fs = 13
    if rmdsi:
        lbl = ['centered on VLFEs','starting 2 min before',
               'ending 2 min after']
        lg=p[0].legend(hh,lbl,loc='upper right',fontsize=fs-1)
        lg.set_title('timing of VLFE interval')

    p[0].set_xlabel('strain rate estimation interval (minutes)',fontsize=fs)
    p[1].set_xlabel('strain rate estimation interval (minutes)',fontsize=fs)
    p[0].set_ylabel('VLFE strain rate / long-term strain rate',fontsize=fs)
    p[1].set_ylabel('excess moment (N m)',fontsize=fs)
    p[1].set_yscale('log')
    xlm = general.minmax(tshorts*1440,1.3)
    xlm = np.exp(general.minmax(np.log(tshorts*1440),1.3))
    xlm[0] = np.minimum(xlm[0],0.5)



    for ph in p:
        ph.set_xlim(xlm)
        ph.set_xscale('log')
        ph.xaxis.set_tick_params(labelsize=fs)
        ph.yaxis.set_tick_params(labelsize=fs)

    
    # second axis
    ylm = np.exp(general.minmax(np.log(np.abs(ylm)),1.2))
    ylm[0] = ylm[0]/1.5
    ylmm = np.log10(ylm/momday)/1.5+mwday
    ax2 = p[1].twinx()
    p[1].set_ylim(ylm)
    ax2.set_ylim(ylmm)
    ax2.set_ylabel('excess moment magnitude',fontsize=fs)
    ax2.yaxis.set_tick_params(labelsize=fs)
    ax2.xaxis.set_tick_params(labelsize=fs)
    
    if fnm:
        nm = 'VLFErtwint_'+fnm
        graphical.printfigure(nm,f)
        plt.close()
