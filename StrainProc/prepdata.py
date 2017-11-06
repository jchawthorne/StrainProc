import os
import numpy.ma as ma
from . import tides
import math
import matplotlib.pyplot as plt
import numpy as np
import obspy
import multiprocessing
from . import fits,readwrite,projcomp


def prepdata(stn,dvar=100.,delfreq=True):

    print(stn)
    # directory to write the fits to
    if delfreq:
        subdir='TIDES-SNR-BOOT'
    else:
        subdir='TIDES-SNR-NOBOOT'
    print(subdir)
    
    # note or set the preferred calibration
    clb=readwrite.defcalib(stn)

    # read the data
    st=readwrite.read(stn,calib=clb)
    
    # remove a long-term trend and exponential
    fpar = {'fitlinear':True,'fitconstant':True,'fitexp':True,
            'flm':np.array([0,float('inf')]),'expdec':300,
            'expdeclim':np.array([30,3000])}
    #fpar['chfit']=['E-N']
    X,Xb=fits.fits.fit(st,fpar=fpar)

    # write fits to a file
    fits.fits.savefits(fpar,X,replace=True,subdir='LONGTERM')
    st,stp=fits.fits.correct(st,fpar,X)

    # fit parameters
    fp = {'fittides':True,'fitatm':True,'delfreq':delfreq,'fitdaily':0,
          'dailyvar':0,'tidespec':'snr','tidepar':0.5,
          'flm':np.array([0.5,6])/86400}

    # daily variation?
    if dvar:
        fp['fitdaily']=1
        fp['dailyvar']=dvar
        
    # fits and corrections
    print('Original fits and corrections')
    fpar=fp.copy()
    X,Xb=fits.fits.fit(st,fpar=fpar)
    stc,stp=fits.fits.correct(st,fpar=fpar,X=X)

    # write fits and corrected data to a file
    fits.fits.savefits(fpar,X,replace=True,subdir=subdir)
    lbl = '-'+str(int(dvar))
    readwrite.writestrain(stc.select(channel='*E*'),'-corr'+lbl)

    # get non-atmospheric components
    print('Identify non-atmospheric components')
    cfn,cfnst,cfnsto = projcomp.nonatm(X)
    sta = projcomp.newch(st,cfnst)+st.select(channel='RDO')

    # write to a file for later
    projcomp.writecf(st[0].stats.station,cfn,subdir)

    # fits for non-atmospheric components
    print('Non-atmospheric fits and corrections')
    fpara=fp.copy()
    fpara['fitatm']=False
    Xa,Xba=fits.fits.fit(sta,fpar=fpara)
    stca,stp=fits.fits.correct(sta,fpar=fpara,X=Xa)
    
    # write fits and corrected data to a file
    fits.fits.savefits(fpara,Xa,replace=True,subdir=subdir)
    lbl = '-'+str(int(dvar))
    readwrite.writestrain(stca.select(channel='*E*'),'-corr'+lbl)

    # write the tidal frequencies used
    fits.fits.writetfreq(fpar['tfreq'],stn,subdir)
    fits.fits.writetfreq(fpara['tfreq'],stn,subdir)

    #print('Plot daily')
    #Xap = {'E-N-na':Xa['E-N-na'],'2EN-na':Xa['2EN-na'],'E+N':X['E+N']}
    #sp.plotdaily(Xap,fparc,st[0].stats.station+lbl)

    print('Finding covariance in corrected components')
    flm=np.array([1./5.,24./6.])/86400.
    chcv=['E-N-na','2EN-na']
    Xcf,frc=projcomp.idcovary(stca,chn=chcv,flm=flm)
    
    # from original coefficients
    cfv=projcomp.multcf(cfn,Xcf)
    stv=projcomp.newch(st,cfv)+st.select(channel='RDO')    
    
    print('Fits for low-covariance components')
    fparv=fp.copy()
    fparv['fitatm']=False
    fparv['chfit']=cfv.keys()
    Xv,Xbv=fits.fits.fit(stv,fpar=fparv)
    stcv,stp=fits.fits.correct(stv,fpar=fparv,X=Xv)

    # write strain, tidal frequencies, coefficients
    fits.fits.savefits(fparv,Xv,replace=True,subdir=subdir)
    readwrite.writestrain(stcv.select(channel='*CV*'),'-corr'+lbl)    
    projcomp.writecf(st[0].stats.station+'-CV',cfv,subdir)
    fits.fits.writetfreq(fparv['tfreq'],stn,subdir)

def runall(stns=None):
    if stns is None:
        # for some stations
        stns = readwrite.localstat(loc='centcasc')
        #stns = readwrite.centcascstat()
        stns.sort()

    # open pool
    p=multiprocessing.Pool(4)
    
    # run
    p.map(prepdata,stns)
