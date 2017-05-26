import os
import numpy as np
import datetime
import math
from scipy.special import sph_harm
import time
import datetime

#dn = np.array([2,0,0,0,0,0])
#dn = np.ndarray(shape=[6,2])
#dn[:,0] = np.array([2,0,0,0,0,0])
#dn[:,1] = np.array([2,1,-1,0,0,0])
#dt = obspy.UTCDateTime('2000-01-01')

#---------------------------------------------------------------
def dnper(dn,dt):
    """
    :param dn:   6-digit Doodson number or a set of them, in 6 by ? numpy array
                   technically Cartwright-Tayler numbers (without the added 5)
    :param dt:   new date (obspy UTCDateTime format)
    :return freqs:   tidal frequencies in 1/day
    :return phs:    phase shifts in degrees
    """

    # phase computation reference date
    dref = datetime.datetime(1900,1,1)

    # date fraction to get to Julian centuries
    # possibly not quite exact, but close enough
    T = dt-dref
    T = T.total_seconds()/86400./36525.

    # frequencies in degrees per hour
    om = np.zeros([7,1])

    # reference phases
    phss = np.zeros([7,1])

    # solar day
    om[0] = 360./24.;
    phss[0] = 0;
    
    # sidereal month---lunar longitude (s)
    om[2] = (481267.89+0.0022*T)/36525./24.
    phss[2] = 277.02 + 481267.89*T + 0.0011*T**2
    
    # tropical year---solar longitude (h)
    om[3] = (36000.77 + 0.0006*T)/36525./24.
    phss[3] = 280.19+36000.77*T+0.0003*T**2

    # lunar day
    om[1] = om[0] - om[2] + om[3]
    phss[1] = 180 - phss[2] + phss[3]

    # lunar perigee (p)
    om[4] = (4069.04-0.0206*T)/36525./24.
    phss[4] = 334.39+4069.04*T-0.0103*T**2

    # lunar nodal regression (N)
    om[5] = -(-1934.14+0.0042*T)/36525./24.
    phss[5] = -(259.16-1934.14*T+0.0021*T**2)
    
    # perihelion (p')
    om[6] = (1.72+0.0010*T)/36525./24.
    phss[6] = 281.22+1.72*T+0.0005*T*2

    # drop the solar day term---not used in multiplication
    om = om[1:]
    phss = phss[1:]

    # change dimensions for multiplication
    om = np.ndarray(shape=[6],buffer=om)
    phss = np.ndarray(shape=[6],buffer=phss)

    # multiply by Doodson number
    freqs = np.dot(dn,om)
    phs = np.dot(dn,phss)
    
    # change to cycles per day
    freqs = freqs * 24. / 360.

    # mod degrees
    phs = np.mod(phs,360)

    # return frequencies and phases
    return freqs,phs


#------------------------------------------------------------------------
def readcte(lat=None,dt=None):
    """
    :param lat:   latitude in degrees
    :param dt:   new date (obspy UTCDateTime format)
    :return tdvl:  dictionary containing
                 amp:   amplitudes for this latitude
                 ampu:   amplitudes without normalization factor
                 freqs:   frequencies in 1/day
                 dnum:   Doodson numbers
                 degs:   degrees for each coefficient
                 dtref:  reference date used (input as dt)
                 lat:    latitude in degrees (input as lat)
    """

    # set a reference time
    if dt is None:
        dt = datetime.datetime(2000,1,1)
    # set a reference latitude
    if lat is None:
        lat = 50

    # read text file with data
    #fdir = os.path.join(os.environ['DATA'],'TIDES')
    fdir,trash = os.path.split(__file__)
    fname = 'Cartwright_Edden_Table.csv'
    fname = os.path.join(fdir,fname)
    vl=np.loadtxt(fname,delimiter=',',dtype=float)

    # Doodson numbers
    dn = vl[:,0:6]

    # amplitudes
    amp = vl[:,6:9]
    amp = np.mean(amp,1)
    amp = amp.astype(complex)

    # degree and order
    n = vl[:,9]
    m = dn[:,0]

    # if m+n is even, cosines
    # if m+n is odd, sines
    sn = np.mod(n+dn[:,0],2)==1
    amp[sn] = amp[sn]*1j

    # grab the relevant periods and phases
    freqs,phs = dnper(dn,dt)

    # modify phases
    phs = np.exp(1j*math.pi/180.*phs)
    amp = np.multiply(amp,phs)

    # copy
    ampu = amp
    
    # add normalizations
    # co-latitude in radians
    colat = (90.-lat)*math.pi/180.

    # unique degrees and orders
    dgs = np.unique(n)
    ods = np.unique(m)
    
    for dg in dgs:
        for od in range(0,int(dg)+1):
            # coefficient
            lgf = sph_harm(od,dg,0,colat)
            
            # because the reference differs from the normalization 
            # for odd degrees
            lgf = lgf * (-1)**od

            # add this normalization to the relevant values
            ii = np.logical_and(m==od,n==dg)
            amp[ii] = amp[ii]*lgf


    # return the set as a dictionary for simplicity
    tdvl = {'amp': amp,'ampu': ampu,'freqs': freqs,'dnum':dn,
            'degs':n,'lat':lat,'dtref':dt}
    return tdvl


def readstraintides(stn):
    """
    :param      stn:  station
    :return   freqs:  frequencies in 1/day
    :return     vls:  values
    :return     stn:  station
    :return reftime:  reference time
    """

    # file name
    mdl = 'gefu-tpxo70'
    fname='strainocean-local-'+mdl+'-plusbody'
    fname=os.path.join(os.environ['DATA'],'TIDES','STRAINCALC',fname)
    

    fl = open(fname,'r')
    line=fl.readline()
    freqs=fl.readline()
    fl.close()
    
    # reference time
    reftime=line.strip()
    reftime=time.strptime(reftime,"%b-%d-%Y %H:%M:%S")
    reftime=time.mktime(reftime)
    reftime=datetime.datetime.fromtimestamp(reftime)

    # frequencies
    freqs=freqs.split()
    freqs=np.array(freqs).astype(float)
    freqs=np.divide(24.,freqs)

    # read the remaining values
    vls = np.loadtxt(fname,dtype=float,skiprows=2)
    
    # locations
    loc = vls[:,0:2]

    # remaining values
    vls = vls[:,2:]

    # separate phases and amplitudes
    ix=np.arange(0,vls.shape[1],2).astype(int)
    amps=vls[:,ix]
    phss=vls[:,ix+1]

    # cosine and sine coefficients
    phss = 1j*math.pi/180.*phss
    phss = np.exp(phss)
    amps = np.multiply(amps,phss)

    # file name
    fname='strainnames'
    fname=os.path.join(os.environ['DATA'],'TIDES','STRAINCALC',fname)

    # names
    nms = np.loadtxt(fname,dtype=str)

    # identify relevant station
    if isinstance(stn,str):
        stn = [stn]
    ix=np.array([])
    for st in stn:
        ixi,=np.where(nms==st)
        ix=np.append(ix,ixi)
    ix=ix.astype(int)
    vls = amps[ix,:]

    # reshape
    typs=np.array(['epsxx,','epsyy','epsxy'])
    Ns,Nf,Nt=len(stn),len(freqs),len(typs)
    vls=vls.reshape([Ns,Nt,Nf])
    vls=np.transpose(vls,[1,2,0])


    # change to desired values
    typs = np.array(['E+N','E-N','2EN'])
    vlsi = vls.copy()
    vlsi[0,:,:]=vls[0,:,:]+vls[1,:,:]
    vlsi[1,:,:]=vls[0,:,:]-vls[1,:,:]
    vlsi[2,:,:]=2*vls[2,:,:]

    vlsi = vlsi*1.e-9

    return freqs,vlsi,stn,reftime,typs
