import os
import obspy
import numpy as np
import importlib
import copy
import pickle

class Fit:
    def __init__(self,name='constant'):
        # define the name
        self.name=name

        # define access to the module
        mdl=importlib.import_module('..fit'+name,package=__name__)

        # set up functions

        # fit preparation
        self.prepfit = mdl.prepfit

        # forward model
        self.formod = mdl.formod

        # prediction
        self.pred = mdl.pred

        # to update
        self.updatepar = mdl.updatepar

        # to update within parameters
        self.calcdpar = mdl.calcdpar
        self.updatedpar = mdl.updatedpar


def strainch():
    """
    :return  chpos:  a list of common channel names, 
                     for deciding values to plot or process
    """
    chpos = ['E+N','E-N','2EN','E+N-na','E-N-na','2EN-na',
             'RS1','RS2','RS3','RS4','HCV','HCV-na','LCV','LCV-na',
             'G0','G1','G2','G3']
    return chpos

def orderedfits():
    """
    :param     psfit:  the possible fits to consider, in order
    """

    psfit=['constant','linear','tides','atm',
           'daily','exp','delresp']

    return psfit


def fit(st,fpar={},starttime=None,endtime=None,flm=None,tdel=[],
        chfit=None,sta=None,psfit=None,**kwargs):
    """
    :param         st: waveforms and supplementary data to fit
    :param       fpar: dictionary of fit parameters---
                        overrides by remaining parameters
              these include the parameters listed here, as well as additional
              possible keyword arguments, which often include fitconstant, 
              fittide, etc, indicating whether those fits should be done
              as well as parameters for individual fits
    :param  starttime: start time  (default: 1.5 years after 
                                       st[0].stats.starttime)
    :param    endtime: end time (default: st[0].stats.endtime)
    :param        flm: filtering  (default:[0,float('inf')]/86400)
    :param       tdel: any intervals to exclude from the fit (default: [])
    :param      chfit: channels to fit (default: everything in st and strainch
    :param        sta: any waveforms to go along with the fit
    :param      psfit: a list of possible fits, in the order the 
                         preparation needs to be done
                      (default: ['constant','linear','tide',
                                 'atm','daily','dailyvar'])
    :param     kwargs: remaining keyword arguments, usually 
                       'fittide','fitconstant',etc
                       may also include parameters to individual fits 
                       not included in fpar,  such as
              dailyvar:   the timescale to allow the linear fit to vary
    :return        Yf: fit results
    """

    # set all the default parameters

    # time limits
    if starttime is None:
        starttime = st[0].stats.starttime + 365.25 * 1.5 * 86400.
    fpar['starttime']=fpar.get('starttime',starttime)
    if endtime is None:
        endtime = st[0].stats.endtime
    fpar['endtime']=fpar.get('endtime',endtime)
    fpar['tdel']=fpar.get('tdel',tdel)

    # filtering
    if flm is None:
        flm = np.array([0,float('inf')])/86400.
    flm=np.atleast_1d(flm)
    fpar['flm']=fpar.get('flm',flm)

    # delete any specified intervals
    if len(fpar['tdel']):
        delints(stf,fpar['tdel'])
        
    # note the possible fits to consider, in order
    if psfit is None:
        psfit=orderedfits()

    # go through and create defaults and classes
    cls=[]
    for ft in psfit:
        # default
        nm='fit'+ft
        fpar[nm]=fpar.get(nm,kwargs.get(nm,0))

        # initialize functions for fit
        if fpar[nm]:
            cls.append(Fit(ft))

    # channels to fit
    if chfit is None:
        # possible channels
        chpos = strainch()
        chfit=np.array([],dtype=str)
        for tr in st:
            if tr.stats.channel in chpos:
                chfit=np.append(chfit,tr.stats.channel)
    fpar['chfit']=fpar.get('chfit',chfit)
    
    # extract data to fit
    # but buffer to avoid edge effects in filter
    tbuf=(1./np.max(flm[flm>0.]))*10.

    # copy and trim everything
    st=st.copy()
    st.trim(starttime=fpar['starttime']-tbuf,
            endtime=fpar['endtime']+tbuf,pad=True)

    # go through and prepare fits    
    if sta is None:
        sta=obspy.Stream()
        for cl in cls:
            sta=sta+cl.prepfit(st,fpar)

    #--------------------prepare the data-------------------------------

    # and select the channels to fit
    stf = obspy.Stream()
    for ch in fpar['chfit']:
        stf=stf+st.select(channel=ch)

    # specify station
    fpar['stn']=fpar.get('stn',stf[0].stats.station)

    # combine and get rid of anything redundant
    stf = stf + sta
    stf = stf.merge()

    # filter
    if fpar['flm'][0]!=0 or fpar['flm'][1]!=float('inf'):
        stf=dtfilt(stf,flm=fpar['flm'])

    # trim
    stf.trim(starttime=fpar['starttime'],endtime=fpar['endtime'],
             pad=True)

    #----------------actually do the inversions---------------------------

    # channel by channel
    chfit = copy.copy(fpar['chfit'])
    Yf,Yfb = {},{}

    for ch in chfit:
        fpar['chfit']=[ch]
        Yfi,Yfbi=compinv(stf=stf,fpar=fpar,cls=cls)   
        Yf[ch],Yfb[ch]=Yfi[ch],Yfbi[ch]
        
    fpar['chfit']=chfit


    return Yf,Yfb


def compinv(stf,fpar,cls,chn=None):
    """
    do the actual inversion
    assumes data has already been selected and filtered
    :param      stf:    waveforms, including data and parameters
    :param     fpar:    fit parameters
    :param      cls:    fits to use
    :param      chn:    channels to consider simultaneously
    :return      Yf:    best-fitting results
    :return     Yfb:    bootstrapped results
    """

    if chn is None:
        chn = copy.copy(fpar['chfit'])
    if isinstance(chn,str):
        chn = [chn]
    fpar['chfit']=chn

    dne = 0
    while dne!=1:

        if dne==0:

            # extract data to match
            data = stf.select(channel=chn[0])[0].data
            
            # create forward models
            nn = stf[0].stats.npts
            mm = 0
            M,ix = [],[]
            for cl in cls:
                # part of the forward matrix
                Mi=cl.formod(stf,fpar)
                M.append(Mi)
                
                # keep track of numbering
                ixi = mm+np.arange(0,M[-1].shape[1])
                ixi=np.atleast_1d(ixi)
                mm=mm+ixi.size
                ix.append(ixi)

        elif dne==-1:

            # if we're updating parameters

            # need a difference
            stc,stp=correct(stf,fpar=fpar,X=Yf,cls=cls)

            # extract data to match
            data = stc.select(channel=chn[0])[0].data

            # create forward models
            nn = stf[0].stats.npts
            mm = 0
            M,ix = [],[]
            for cl in cls:
                # part of the forward matrix
                Mi=cl.calcdpar(stf,fpar=fpar,X=Yf)
                M.append(Mi)
                
                # keep track of numbering
                ixi = mm+np.arange(0,M[-1].shape[1])
                ixi=np.atleast_1d(ixi)
                mm=mm+ixi.size
                ix.append(ixi)

        # complete the matrix
        Mmat=np.ndarray([nn,mm])
        Mmat=np.ma.masked_array(Mmat,mask=False)
        for k in range(0,len(ix)):
            for m in range(0,ix[k].size):
                Mmat[:,ix[k][m]]=M[k][:,m]
                Mmat=np.atleast_2d(Mmat)
                    
        # choose a mask from observations
        iok = np.ones(nn,dtype=bool)
        if isinstance(data,np.ma.masked_array):
            iok = np.logical_and(~data.mask,iok)
            iok = np.logical_and(iok,~np.isnan(data.data))
        else:
            iok = np.logical_and(iok,~np.isnan(data))

        # choose a mask from forward model
        prd = np.sum(Mmat,1)
        if isinstance(prd,np.ma.masked_array):
            msk = ~np.sum(Mmat.mask,axis=1).astype(bool)
            iok = np.logical_and(iok,msk)
            iok = np.logical_and(iok,~np.isnan(prd))
        else:
            iok = np.logical_and(iok,~np.isnan(prd))

        # if there's enough data
        iok,=np.where(iok)

        if len(iok)>mm:
            # split some values
            Mmat = Mmat[iok,:]
            data = data[iok]

            # this fit
            [Y,rsd,rank,s]=np.linalg.lstsq(Mmat,data)
        
        else:
            
            # fits
            Y = np.ones(mm)*float('nan')

        # split the different fits
        if dne>-1:
            Yf={}
            for k in range(0,len(ix)):
                Yf[cls[k].name]=Y[ix[k]]

            # initialize bootstrap results
            Nboot,Yfb=10,{}
            for k in range(0,len(ix)):
                ni = len(ix[k])
                Yfb[cls[k].name]=np.full([ni,Nboot],np.nan)

            if len(iok)>mm:

                Ni=100
                ixb=np.linspace(0.,len(iok)-1,Ni+1).astype(int)
                for m in range(0,Nboot):
                    # indices
                    ixi=(np.random.rand(Ni)*Ni).astype(int)
                    iget=np.array([])
                    for ixj in ixi:
                        iadd=np.arange(ixb[ixj],ixb[ixj+1],1)
                        iget=np.append(iget,iadd)
                        iget=iget.astype(int)
                        
                    # this fit
                    [Y,rsd,rank,s]=np.linalg.lstsq(Mmat[iget,:],data[iget])
            
                    # save results
                    for k in range(0,len(ix)):
                        Yfb[cls[k].name][:,m]=Y[ix[k]]

            # dictionary for components
            Yf = {chn[0]:Yf}
            Yfb = {chn[0]:Yfb}

            # go through and check for updates
            dne = 1
            for cl in cls:
                dnei=cl.updatepar(st=stf,fpar=fpar,X=Yf,
                                  Xb=Yfb,sta=stf)
                dne=np.minimum(dne,int(dnei))

        else: 
            # split the different updates
            Yfdiff={}
            for k in range(0,len(ix)):
                Yfdiff[cls[k].name]=Y[ix[k]]

            # dictionary for components
            Yfdiff = {chn[0]:Yfdiff}
            
            # change preferred parameters
            for cl in cls:
                cl.updatedpar(st=stf,fpar=fpar,X=Yf,Xdiff=Yfdiff)

            dne = 0


    return Yf,Yfb


def correct(st,fpar,X,psfit=None,cls=None):
    """
    :param    st:  waveforms
    :param  fpar:  fit parameters
    :param     X:  fit results
    :param psfit:  possible fits to consider, in order
                    (default: as from orderfits)
    :param   cls:  fits, if already known (default: calculated)
    :return  stc:  corrected data
    :return  stp:  predictions
    """


    # go through and create defaults and classes
    if cls is None:
        # note the possible fits to consider, in order
        if psfit is None:
            psfit=orderedfits()

        cls=[]
        for ft in psfit:
            # default
            nm='fit'+ft
            fpar[nm]=fpar.get(nm,0)
            
            # initialize functions for fit
            if fpar[nm]:
                cls.append(Fit(ft))
    
    # initialize predictions
    stp = obspy.Stream()

    # merge for convenience and create a copy for corrections
    stc=st.merge().copy()

    for ch in fpar['chfit']:
        sti=stc.select(channel=ch)
        if sti:
            stpi=sti.copy()
            stpi[0].data[:]=0.
            for cl in cls:
                # prediction for each fit
                prd=cl.pred(sti,fpar=fpar,X=X[ch],sta=st)

                # add to set
                stpi[0].data=stpi[0].data+prd

            # subtract
            sti[0].data=sti[0].data-stpi[0].data
            stp=stp+stpi

    return stc,stp


def writetfreq(tfreq,stn):
    """
    :param  tfreq:  dictionary of frequencies used
    :param    stn:  station
    """

    # directory
    fdir = os.path.join(os.environ['DATA'],'STRAINPROC','ATMCF')
    for ch in tfreq.keys():
        fname = 'tfreq_'+stn+'_'+ch
        fl = open(os.path.join(fdir,fname),'w')
        for vl in tfreq[ch]:
            fl.write(str(vl))
            fl.write('\n')
        fl.close()


def dtfilt(st,flm=None,taplen=None):
    """
    filters the stream object, detrending and tapering as necessary
    :param     st: waveforms
    :param    flm: frequencies for filtering (default: 1-10)
    :param taplen: max taper length (default: 2/flm[0])
    :return    st: modified traces
    """
    if flm is None:
        flm=np.array([1,10])
    if taplen is None:
        taplen=2./flm[0]

    # detrend and taper---only for high-pass and bandpass?
    if (flm[0]>0):
        st=st.split().detrend()
        st=st.taper(type='cosine',max_percentage=0.5,max_length=taplen)

    # filter
    if (flm[0]>0) and (flm[1]<float('inf')):
        st=st.filter('bandpass',freqmin=flm[0],freqmax=flm[1],
                       corners=1,zerophase=True)
    elif flm[0]>0:
        st=st.filter('highpass',freq=flm[0],corners=1,zerophase=True)
    elif flm[1]<float('inf'):
        st=st.filter('lowpass',freq=flm[1],corners=1,zerophase=True)
                       
    # merge to resample
    st.merge(fill_value=None)

    return st


def savefits(fpar,X,subdir='DEFAULT',replace=False):
    """
    :param     fpar:   fit parameters 
    :param        X:   fit results
    :param   subdir:   directory to save to, within $DATA/STRAINPROC/SAVEDFITS
    :param  replace:   replace previous file?  (default: False)
    """

    # directory
    fdir=os.path.join(os.environ['DATA'],'STRAINPROC','SAVEDFITS',subdir)
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # station
    try:
        stn = fpar['stn']
    except:
        stn = 'unknown'

    # save each channel separately
    chn=copy.copy(fpar['chfit'])
    chn=copy.copy(X.keys())

    for ch in chn:
        # complete file name
        fnamefpar=os.path.join(fdir,stn+'-'+ch+'-fpar')
        fnameX=os.path.join(fdir,stn+'-'+ch+'-X')
        
        # if we don't want to overwrite previous files
        if os.path.exists(fnamefpar) or os.path.exists(fnameX):
            if not replace:
                k=1
                fnamefpari,fnameXi=fnamefpar+'-'+str(k),fnameX+'-'+str(k)
                while os.path.exists(fnamefpari) or os.path.exists(fnameXi):
                    k=k+1
                    fnamefpari,fnameXi=fnamefpar+'-'+str(k),fnameX+'-'+str(k)
                fnamefpar,fnameX=fnamefpari,fnameXi
            
        # write to files
        with open(fnamefpar,'w') as fl:
            pickle.dump(fpar,fl)
        with open(fnameX,'w') as fl:
            pickle.dump(X[ch],fl)

            
def loadfits(stn='unknown',chn=None,subdir='DEFAULT',sfx=''):
    """
    :param      stn:   station name
    :param      chn:   channels to get (default: ['E+N','E-N','2EN'])
    :param   subdir:   directory to save to, within $DATA/STRAINPROC/SAVEDFITS
    :param      sfx:   suffix, such as '-1' in case it's a later version
    :return    fpar:   the input fit parameters
    :return       X:   the resulting fit parameters
    """

    # directory
    fdir=os.path.join(os.environ['DATA'],'STRAINPROC','SAVEDFITS',subdir)

    if chn is None:
        chn = ['E+N','E-N','2EN']

    # initialize
    X = {}
    
    for ch in chn:
        # complete file name
        fnamefpar=os.path.join(fdir,stn+'-'+ch+'-fpar'+sfx)
        fnameX=os.path.join(fdir,stn+'-'+ch+'-X'+sfx)

        with open(fnamefpar,'r') as fl:
            # just keep replacing fpar
            fpar=pickle.load(fl)
        with open(fnameX,'r') as fl:
            X[ch]=pickle.load(fl)

    # replace the channels
    fpar['chfit']=chn

    # replace the frequencies and decay constants if desired
    fpar['tfreq']={}
    fpar['expdec']={}
    for ch in chn:
        if 'tfreq' in X[ch].keys():
            fpar['tfreq'][ch] = X[ch]['tfreq']
        if 'expdec' in X[ch].keys():
            fpar['expdec'][ch] = X[ch]['expdec']
    

    return fpar,X


def corrfromfile(st,chn=None,subdir='DEFAULT'):
    """
    :param      st:  strain data
    :param     chn:  channels to get 
                     (default: those in waveforms and strainch)
    :param  subdir:  directory containing data, within
                        $DATA/STRAINPROC/SAVEDFITS
    :return    stc:  corrected data
    :return    stp:  predictions that were removed
    :return   fpar:  the fit parameters
    :return      X:  the fit results
    """

    if chn is None:
        chpos = strainch()
        chn=np.array([],dtype=str)
        for tr in st:
            if tr.stats.channel in chpos:
                chn=np.append(chn,tr.stats.channel)    

    # load the results
    fpar,X=loadfits(stn=st[0].stats.station,chn=chn,subdir=subdir)

    # compute corrections
    stc,stp=correct(st=st,fpar=fpar,X=X)

    return stc,stp,fpar,X
