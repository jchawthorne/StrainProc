import numpy as np
import os
import math
import matplotlib.pyplot as plt
import graphical

def plotmodprof():
    
    plt.close()
    f = plt.figure()
    
    p = plt.axes([.1,.1,.85,.85])

    
    # depths
    x = np.arange(-100.,100,0.1)

    # frequencies
    freq = np.fft.fftfreq(len(x),np.diff(x)[0])
    k = 2*math.pi*freq

    mvar = np.array([1.,0.5,0.1])
    
    zs = 6.
    s = 1.
    HH = [2,2,0.1]
    dz = 0.1

    vls = np.ndarray([len(x),len(mvar)],dtype=complex)

    for m in range(0,len(mvar)):
        H = HH[m]
        cf = -1j * s * dz * np.sign(k) 
        cf = np.multiply(cf,1j*k)
        cf1 = np.exp((H-zs)*np.abs(k))
        cf2 = np.cosh(H*np.abs(k)) + mvar[m] * np.sinh(H*np.abs(k))
        cf = np.multiply(cf,cf1)
        cf = np.divide(cf,cf2)
        
        vls[:,m] = cf

    vls = np.fft.ifft(vls,axis=0).astype(float)
    ishf = np.argmin(np.abs(x))
    vls = np.roll(vls,ishf,axis=0)
    nm = np.max(np.abs(vls[:,0]))

    h,lbl = [],[]
    for m in range(0,len(mvar)):
        hh, = plt.plot(x,vls[:,m]/nm)
        h.append(hh)
        if mvar[m]==1:
            lbli = 'homogeneous half space'
        else:
            lbli = str(HH[m])+' km  with $\mu=$' + str(mvar[m]) + ' $\mu_{ref}$'
        lbl.append(lbli)

    plt.xlabel('fault-perpendicular distance (km)')
    plt.ylabel('simple shear strain, normalized by half space maximum')
    plt.xlim([-50,50])

    lg=plt.legend(h,lbl,fontsize='small')

    graphical.printfigure('SEQstrainpossible',f)

    return vls,x
        
def readlinetal():
    """
    :return     loc:  locations--lon,lat,depth
    :return    pvel:  P-wave velocities
    """
    fname=os.path.join(os.environ['DATA'],'VELMODELS','Linetal_2010.vp')
    
    # read
    vls=np.loadtxt(fname,dtype=float,skiprows=1)

    # location
    loc = vls[:,0:3]

    # p-wave velocities
    pvel = vls[:,5]

    return loc,pvel

def avedepthprof(mloc=[-121.55,36.84],hwidth=10.,loc=None,pvel=None):
    """
    :param     mloc: location of interest
    :param   hwidth: half-width for smoothing in km
    :param      loc:  locations--lon,lat,depth
    :param     pvel:  P-wave velocities
    :return   pvela:  P-velocity depth profile
    :return     dep:  depths in km
    """

    # read the velocity model if not given
    if loc is None:
        loc,pvel=readlinetal()

    # horizontal distances to location
    rlon = math.cos(math.pi/180.*mloc[1])
    dst = np.power(loc[:,1]-mloc[1],2) 
    dst = dst+rlon**2*np.power(loc[:,0]-mloc[0],2) 
    dst = np.power(dst,0.5)*110.567

    # weights
    wgt = np.exp(-np.power(dst,2)/hwidth**2)

    # depths
    dep = np.unique(loc[:,2])
    pvela = np.ndarray(dep.shape)
    
    for k in range(0,len(dep)):

        # relevant depth
        ii = loc[:,2]==dep[k]

        # change weighting
        wgts = wgt[ii]/np.sum(wgt[ii])

        # average
        pvela[k]=np.dot(wgts,pvel[ii])
    

    return pvela,dep
