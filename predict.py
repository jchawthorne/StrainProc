import okada_wrapper as ow
import numpy as np
import math

def teststrainpoint():
    import string
    # calculations to reproduce table 2 of Okada (1985)

    x=[2,2,0,0]
    y=[3,3,0,0]
    d=[4,4,4,4]
    # note these are positive up dips
    dip=[70,70,90,-90]
    L=[0,3,3,3]
    W=[0,2,2,2]

    for k in range(0,len(x)):
        print('Case '+str(k+1))
        xi,yi,di,dipi,Li,Wi=x[k],y[k],d[k],-dip[k],L[k],W[k]
        strike=90.

        if Li>0:
            potency = Li*Wi*1.
            Li = [0,Li]
            Wi = [0,Wi]
        else:
            potency = 1.

        lbl='strike'
        u,grad_u=defcalc(pr=0.25,locobs_m=[xi,yi],strike=strike,
                         depth=di,dip=dipi,potency=potency,rake=0,
                         L=Li,W=Wi)
        u = np.round(u,10)
        grad_u = np.round(grad_u,10)

        line = ["{0:0.3e}".format(vl) for vl in u]
        line.append("{0:0.3e}".format(grad_u[0,0]))
        line.append("{0:0.3e}".format(grad_u[1,0]))
        line.append("{0:0.3e}".format(grad_u[0,1]))
        line.append("{0:0.3e}".format(grad_u[1,1]))
        line.append("{0:0.3e}".format(grad_u[0,2]))
        line.append("{0:0.3e}".format(grad_u[1,2]))
        line=[vl.rjust(12) for vl in line]
        print(lbl.ljust(12)+string.join(line))
        
        lbl='dip'
        u,grad_u=defcalc(pr=0.25,locobs_m=[xi,yi],strike=strike,
                         depth=di,dip=dipi,potency=potency,rake=90,
                         L=Li,W=Wi)
        u = np.round(u,10)
        grad_u = np.round(grad_u,10)
        
        line = ["{0:0.3e}".format(vl) for vl in u]
        line.append("{0:0.3e}".format(grad_u[0,0]))
        line.append("{0:0.3e}".format(grad_u[1,0]))
        line.append("{0:0.3e}".format(grad_u[0,1]))
        line.append("{0:0.3e}".format(grad_u[1,1]))
        line.append("{0:0.3e}".format(grad_u[0,2]))
        line.append("{0:0.3e}".format(grad_u[1,2]))
        line=[vl.rjust(12) for vl in line]
        print(lbl.ljust(12)+string.join(line))
        
        
        lbl='tensile'
        u,grad_u=defcalc(pr=0.25,locobs_m=[xi,yi],strike=strike,
                         depth=di,dip=dipi,potency=potency,rake=90,
                         tensilefrac=1.,L=Li,W=Wi)
        u = np.round(u,10)
        grad_u = np.round(grad_u,10)
        
        line = ["{0:0.3e}".format(vl) for vl in u]
        line.append("{0:0.3e}".format(grad_u[0,0]))
        line.append("{0:0.3e}".format(grad_u[1,0]))
        line.append("{0:0.3e}".format(grad_u[0,1]))
        line.append("{0:0.3e}".format(grad_u[1,1]))
        line.append("{0:0.3e}".format(grad_u[0,2]))
        line.append("{0:0.3e}".format(grad_u[1,2]))
        line=[vl.rjust(12) for vl in line]
        print(lbl.ljust(12)+string.join(line))
    
        

def calcstrain(cmps=None,st=None,*args,**kwargs):
    """
    calculate strain at a point of interest
    :param      cmps: strain components of interest 
                    (default: ['E-N','2EN','E+N','ZZ','2EZ','2NZ'])
    :param        st: waveform with strain data,
                      with location in st[0].stats.sac.stlo/a
    :remaining parameters as input to defcalc:
    """

    if cmps is None:
        cmps = ['E-N','2EN','E+N','ZZ','2EZ','2NZ']

    if st is not None:
        # set channel from the traces
        import obspy
        if isinstance(st,obspy.Stream):
            cmps=[tr.stats.channel for tr in st]
            st=st[0]
        else:
            cmps=[tr.stats.channel]

        # as well as the location if not given
        if (not 'locobs_deg' in kwargs.keys() and 
            not 'locobs_m' in kwargs.keys()):
            kwargs['locobs_deg']=[st.stats.sac.stlo,st.stats.sac.stla]


    # calculate the displacment and gradient
    u,grad_u = defcalc(*args,**kwargs)

    # map to strain
    epsmat = (grad_u + grad_u.transpose())/2.

    # and extract the desired components
    strain,cmps=epsmat2scomp(epsmat,cmps)

    return strain,cmps


def epsmat2scomp(epsmat,cmps):
    """
    :param    epsmat: matrix with strains, usually, E,N,up in columns/rows
    :param      cmps: strain components to extract
    :param    strain: the desired strains
    """

    # map east to x, north to y
    cmpsi=[sti.replace('E','X') for sti in cmps]
    cmpsi=[sti.replace('N','Y') for sti in cmpsi]
    
    # no double letters---for extension
    cmpsi=[sti.replace('XX','X') for sti in cmpsi]
    cmpsi=[sti.replace('YY','Y') for sti in cmpsi]
    cmpsi=[sti.replace('ZZ','Z') for sti in cmpsi]

    # capitalize
    cmpsi=[sti.upper() for sti in cmpsi]

    # the mappings
    ii={'X':[0,0],'Y':[1,1],'Z':[2,2],'XY':[0,1],'XZ':[0,2],'YZ':[1,2]}

    # allow a third dimension for more values
    if epsmat.ndim==2:
        justone=True
        epsmat=epsmat.reshape([epsmat.shape[0],epsmat.shape[1],1])
    else:
        justone=False
    
    # collect the strain
    strain = []
    
    for cm in cmpsi:
        # extract the strain component of interest
        if '-' in cm:
            # if it's a difference
            vl = cm.split('-')
            ix1,ix2 = ii.get(vl[0]),ii.get(vl[1])
            straini = epsmat[ix1[0],ix1[1],:] - epsmat[ix2[0],ix2[1],:]
        elif '+' in cm:
            # if it's a sum
            vl = cm.split('+')
            ix1,ix2 = ii.get(vl[0]),ii.get(vl[1])
            straini = epsmat[ix1[0],ix1[1],:] + epsmat[ix2[0],ix2[1],:]
        elif '2' in cm:
            # if it's something doubled
            ix1 = ii.get(cm[1:])
            straini = 2.*epsmat[ix1[0],ix1[1],:]
        else:
            # should just be one value
            ix1 = ii.get(cm)
            straini = epsmat[ix1[0],ix1[1],:]
        
        # add these to the set
        strain.append(straini)
            
    # create a matrix
    strain=np.vstack(strain)

    # or an array if there was already one
    if justone:
        strain=strain.flatten()

    return strain,cmps
    


def defcalc(shmod=3.e10,pr=0.25,
            locobs_m=None,locslip_m=None,
            locobs_deg=None,locslip_deg=None,
            depth=5.,dip=45.,strike=0.,
            potency=1.,moment=None,meanslip=None,rake=90.,
            tensilefrac=0.,L=0,W=0):
    """
    :param       shmod:  shear modulus (default: 3e10)
                           just for converting moment to potency
    :param          pr:  Poisson's ratio (default: 0.25)
    :param    locobs_m:  location of the observation, in meters [east,north]
    :param   locslip_m:  location of the slip, in meters [east,north]
    :param  locobs_deg:  location of the observation, in meters [east,north]
    :param locslip_deg:  location of the slip, in meters [east,north]
    :param       depth:  depth of slip, in km
    :param         dip:  dip, in degrees positive down, dip to the right of strike
                            (default: 45)
    :param     potency:  potency in m^3 (default: 1)
    :param      moment:  moment, in N-m, overrides potency if given
    :param    meanslip:  average slip in m, overrides potency and moment if given
    :param      strike:  strike in degrees E of N (default: 0)
    :param        rake:  rake in degrees (default: 90)
                              (0: left-lateral, 90: thrust, -90: normal)
    :param tensilefrac:  portion of the "moment" that is tensile opening
                              (default: 0)
    :param           L:  along-strike length centered at locslip,
                              or a vector from that point(default: 0)
    :param           W:  along-dip length centered at locslip,
                              or a vector from that point, 
                              with positive down-dip (default: 0)
    :return          u:  displacements ([E,N,up])
    :return     grad_u:  displacement gradients  ([E,N,up])
                          columns (2nd dim) index displacement directions
                          rows (1st dim) index derivatives, so that
                              dux / dy is in grad_u[1,0]
    """

    # to lengths
    try:
        Lv = L
        L = np.diff(L)
    except:
        Lv = [-.5*L,.5*L]
    try:
        Wv = W
        W = np.diff(W)
    except:
        Wv = [-.5*W,.5*W]

    # to the Lame parameter
    # and relative deformation
    lda = 2.*pr/(1.-2.*pr)
    alpha = (lda + 1.) / (lda + 2. * 1.)

    # potency
    if moment is not None:
        potency = moment / shmod

    if meanslip is not None:
        potency = meanslip * L * W

    # determine observation relative locations in m
    if locobs_m is not None:
        locobs = locobs_m
    elif locobs_deg is not None:
        rlon = math.cos(locobs_deg[1]*math.pi/180)
        if locslip_deg is not None:
            locobs_deg=np.atleast_1d(locobs_deg)-np.atleast_1d(locslip_deg)
            locslip_deg=[0.,0.]
        locobs = [locobs_deg[0]*rlon*111.,locobs_deg[1]*111.]
        locobs = np.array(locobs)*1000
    else:
        locobs = [0.,0.]

    # determine relative locations in m
    if locslip_m is not None:
        locslip = locslip_m
    elif locslip_deg is not None:
        rlon = math.cos(locslip_deg[1]*math.pi/180)
        locslip = [locslip_deg[0]*rlon*111.,locslip_deg[1]*111.]
        locslip = np.array(locslip)*1000
    else:
        locslip = [0.,0.]

    # location observation point relative to slip location
    enobs = np.array(locobs)-np.array(locslip)

    
    # project to the fault plane along x,
    # counterclockwise from fault plane along y
    cs,ss=math.cos(strike*math.pi/180),math.sin(strike*math.pi/180)
    xobs = enobs[0]*ss+enobs[1]*cs
    yobs = -enobs[0]*cs+enobs[1]*ss

    xo = [xobs,yobs,0.]

    # dip-slip portion
    potdip = potency*math.sin(rake*math.pi/180.)*(1-tensilefrac)
    # strike-slip portion
    potss = potency*math.cos(rake*math.pi/180.)*(1-tensilefrac)
    # tensile potency
    pottens=potency*tensilefrac

    # compute
    if L==0:
        # for input to a point source
        pot = [potss,potdip,pottens,0.]
        success,u,grad_u=ow.dc3d0wrapper(alpha,xo,depth,dip,pot)
    else:
        # for a finite plane
        disl=[potss/(L*W),potdip/(L*W),pottens/(L*W)]
        success,u,grad_u=ow.dc3dwrapper(alpha,xo,depth,dip,Lv,Wv,disl)

    # need to rotate back
    R = np.vstack([[ss,-cs,0],[cs,ss,0],[0,0,1]])
    grad_u = np.matmul(R,grad_u)
    grad_u = np.matmul(grad_u,R.transpose())
    u = np.matmul(R,u)

    return u,grad_u
