import Strain
import tides
import numpy as np
import os
import obspy

def calibdata(stn='B014',cname='Hawthorne2'):
    """
    :param    stn:  station
    :param  cname:  directory to save the results in
    :return    Ci:  coupling matrix
    :return   och:  observation channels
    :return   pch:  resulting channels
    """

    # read the data
    st=Strain.readwrite.read(stn,calib=None)


    # to print
    fname=os.path.join(os.environ['DATA'],'STRAINPROC','CALIBMAT',
                       cname,'preferred_gages')
    fl = open(fname,'r')
    chfit=['G0','G1','G2','G3']
    for line in fl:
        vls = line.split()
        if vls[0]==stn:
            chfit = vls[1].split(',')
    fl.close()

    # read the tides
    freqs,vls,stns,reftime,typs=tides.readstraintides(stn)

    # parameters
    fpar = {'fittides':1,'tidespec':'5big',
            'chfit':chfit,
            'flm':np.array([1./3.,float('inf')])/86400.}
    fpar['dtr']=obspy.UTCDateTime(reftime)
    fpar['starttime']=st[0].stats.starttime+365.25*1.5*86400.
    fpar['endtime']=st[0].stats.endtime
    #fpar['endtime']=fpar['starttime']+100*86400
    fpar['fitatm']=1

    # fit the data
    X,Xb=Strain.fits.fit(st,fpar=fpar)

    # frequencies
    fpick=np.array([12.4206,25.8193])
    fpick=np.divide(24.,fpick)
    
    # indices of the tides in the fit and predictions
    i1,i2=np.array([]),np.array([])
    fobs=fpar['tfreq'][fpar['chfit'][0]]
    for fpicki in fpick:
        iadd=np.argmin(np.abs(fobs-fpicki))
        i1=np.append(i1,iadd)
        iadd=np.argmin(np.abs(freqs-fpicki))
        i2=np.append(i2,iadd)
    i1,i2=i1.astype(int),i2.astype(int)

    # predicted
    P=np.ndarray([len(fpick)*2,len(typs)])
    for k in range(0,len(fpick)):
        P[k*2,:]=np.real(vls[:,i1[k],:]).flatten()
        P[k*2+1,:]=np.imag(vls[:,i1[k],:]).flatten()

    # observed
    O=np.ndarray([len(fpick)*2,len(fpar['chfit'])])
    for m in range(0,len(fpar['chfit'])):
        chi=fpar['chfit'][m]
        vli=X[chi]['tides']
        for k in range(0,len(fpick)):
            O[k*2,m]=vli[i2[k]*2]
            O[k*2+1,m]=vli[i2[k]*2+1]

    # solve---mapping from areal/shear to gauges
    [C,rsd,rank,s]=np.linalg.lstsq(P,O)
    C = C.transpose()

    # pseudoinverse---mapping from gauges to areal/shear
    Ci = np.linalg.pinv(C)

    # to return
    och = np.atleast_1d(fpar['chfit'])
    pch = typs

    # to print
    fname=os.path.join(os.environ['DATA'],'STRAINPROC','CALIBMAT',
                       cname,stn)
    fl = open(fname,'w')
    fl.write(','.join(och)+'\n')
    fl.write(','.join(pch)+'\n')
    fl.write(stn+'\t')
    for vl in Ci.flatten():
        fl.write(str(vl)+'\t')
    fl.close()

    return Ci,och,pch

    
