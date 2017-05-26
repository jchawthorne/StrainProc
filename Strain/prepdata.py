import os
import numpy.ma as ma
import tides
import math
import matplotlib.pyplot as plt
import numpy as np
import obspy
import multiprocessing
import Strain

def prepdata(stn,dvar=100.):

    print(stn)
    # directory to write the fits to
    subdir='TIDES-SNR'
    
    # note or set the preferred calibration
    clb=Strain.readwrite.defcalib(stn)

    # read the data
    st=Strain.readwrite.read(stn,calib=clb)
    
    # remove a long-term trend and exponential
    fpar = {'fitlinear':True,'fitconstant':True,'fitexp':True,
            'flm':np.array([0,float('inf')]),'expdec':300,
            'expdeclim':np.array([30,3000])}
    #fpar['chfit']=['E-N']
    X,Xb=Strain.fits.fits.fit(st,fpar=fpar)

    # write fits to a file
    Strain.fits.fits.savefits(fpar,X,replace=True,subdir='LONGTERM')
    st,stp=Strain.fits.fits.correct(st,fpar,X)

    # fit parameters
    fp = {'fittides':True,'fitatm':True,'delfreq':True,'fitdaily':0,
          'fitdailyvar':0,'tidespec':'snr','tidepar':0.5,
          'flm':np.array([0.5,6])/86400}

    # daily variation?
    if dvar:
        fp['fitdaily']=1
        fp['fitdailyvar']=dvar
        
    # fits and corrections
    print('Original fits and corrections')
    fpar=fp.copy()
    X,Xb=Strain.fits.fits.fit(st,fpar=fpar)
    stc,stp=Strain.fits.fits.correct(st,fpar=fpar,X=X)

    # write fits and corrected data to a file
    Strain.fits.fits.savefits(fpar,X,replace=True,subdir=subdir)
    lbl = '-'+str(int(dvar))
    Strain.readwrite.writestrain(stc.select(channel='*E*'),'-corr'+lbl)

    # get non-atmospheric components
    print('Identify non-atmospheric components')
    cfn,cfnst,cfnsto = Strain.projcomp.nonatm(X)
    sta = Strain.projcomp.newch(st,cfnst)+st.select(channel='RDO')

    # write to a file for later
    Strain.projcomp.writecf(st[0].stats.station,cfn)

    # fits for non-atmospheric components
    print('Non-atmospheric fits and corrections')
    fpara=fp.copy()
    fpara['fitatm']=False
    Xa,Xba=Strain.fits.fits.fit(sta,fpar=fpara)
    stca,stp=Strain.fits.fits.correct(sta,fpar=fpara,X=Xa)
    
    # write fits and corrected data to a file
    Strain.fits.fits.savefits(fpara,Xa,replace=True,subdir=subdir)
    lbl = '-'+str(int(dvar))
    Strain.readwrite.writestrain(stca.select(channel='*E*'),'-corr'+lbl)

    # write the tidal frequencies used
    Strain.fits.fits.writetfreq(fpar['tfreq'],stn)
    Strain.fits.fits.writetfreq(fpara['tfreq'],stn)

    #print('Plot daily')
    #Xap = {'E-N-na':Xa['E-N-na'],'2EN-na':Xa['2EN-na'],'E+N':X['E+N']}
    #sp.plotdaily(Xap,fparc,st[0].stats.station+lbl)

    print('Finding covariance in corrected components')
    flm=np.array([1./5.,24./6.])/86400.
    chcv=['E-N-na','2EN-na']
    Xcf,frc=Strain.projcomp.idcovary(stca,chn=chcv,flm=flm)
    
    # from original coefficients
    cfv=Strain.projcomp.multcf(cfn,Xcf)
    stv=Strain.projcomp.newch(st,cfv)+st.select(channel='RDO')    
    
    print('Fits for low-covariance components')
    fparv=fp.copy()
    fparv['fitatm']=False
    fparv['chfit']=cfv.keys()
    Xv,Xbv=Strain.fits.fits.fit(stv,fpar=fparv)
    stcv,stp=Strain.fits.fits.correct(stv,fpar=fparv,X=Xv)

    # write strain, tidal frequencies, coefficients
    Strain.fits.fits.savefits(fparv,Xv,replace=True,subdir=subdir)
    Strain.readwrite.writestrain(stcv.select(channel='*CV*'),'-corr'+lbl)    
    Strain.projcomp.writecf(st[0].stats.station+'-CV',cfv)
    Strain.fits.fits.writetfreq(fparv['tfreq'],stn)

def runall(stns=None):
    if stns is None:
        # for some stations
        stns = Strain.readwrite.centcascstat()
        stns.sort()

    # open pool
    p=multiprocessing.Pool(4)
    
    # run
    p.map(prepdata,stns)
