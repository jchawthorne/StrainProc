import matplotlib.pyplot as plt
import general
import numpy as np
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.backends.backend_pdf import PdfPages
import os
from scipy.interpolate import interp1d

def minmax(x,bfr=1.):
    """
    :param     x:  values
    :param   bfr:  buffer
    :return   lm:  limits
    """

    # limits
    lm=np.array([np.min(x),np.max(x)])
    df=(1-bfr)*np.diff(lm)[0]/2.*np.array([-1.,1])
    lm=lm+df

    return lm


def printfigure(fname,f,ftype='pdf'):
    """
    :param     fname:   file name
    :param         f:   figure handle
    :param     ftype:   figure type (default: 'pdf')
    """

    fname=os.path.join(os.environ['FIGURES'],fname+'.'+ftype)
    if ftype=='pdf':
        pp=PdfPages(fname)
        pp.savefig(f)
        pp.close()
    else:
        f.savefig(fname)
        

def baroutlinevals(x,y,wzeros=False):
    """
    :param      x:  x-values: center or edges of a histogram
    :param      y:  number of values in each bin
    :param wzeros:  include zero-values at the end to 
                       create a closed polygon (default: False)
    :param     xl:  x-locations of the edges
    :param     yl:  y-locations of the edges
    """

    x=np.atleast_1d(x)
    y=np.atleast_1d(y)

    # number of bins
    N = y.size
    
    if x.size==N:
        # for bin centers
        dx = np.median(np.diff(x))
        xc = (x[0:-1]+x[1:])/2.
        xc=np.append(np.array(x[0]-dx/2.),xc)
        x=np.append(xc,np.array(x[-1]+dx/2.))
    elif x.size==N+1:
        # for bin edges
        pass
    else:
        print('Wrong number of bins')

    # which indices
    ix=np.repeat(np.arange(0,N+1),2)
    ix=ix[1:-1]

    iy=np.repeat(np.arange(0,N),2)

    # extract the limits
    xl = x[ix]
    yl = y[iy]

    # add points at zero
    if wzeros:
        xl=np.hstack([xl[0:1],xl,xl[-1:],xl[0:1]])
        yl=np.hstack([np.zeros(1),yl,np.zeros(2)])

    return xl,yl



def colors(N,lgt=False):
    """
    :param      N:  number of colors
    :param    lgt:  lighter colors (default: False)
    :return  cols:  colors
    """
    
    if not lgt:
        red = 'firebrick'
        blue =  'darkblue'
        yellow = 'darkgoldenrod'
        green = 'darkgreen'
        pink = 'deeppink'
        violet = 'darkviolet'
        orange = 'orange'
    else:
        red = 'darksalmon'
        blue =  'lightsteelblue'
        yellow = 'palegoldenrod'
        green = 'lightsage'
        pink = 'lightpink'
        violet = 'thistle'
        orange = 'navajowhite'

    if N==0:
        cols = []
    elif N==1:
        cols = [red]
    elif N==2:
        cols = [red,blue]
    elif N==3:
        cols = [red,yellow,blue]
    elif N==4:
        cols = [red,yellow,green,blue]
    elif N==5:
        cols = [red,orange,yellow,green,blue]
    elif N==6:
        cols = [red,orange,yellow,green,blue,violet]
    elif N==7:
        cols = [pink,red,orange,yellow,green,blue,violet]
    elif N>7:
        # original map
        colsi = colors(7,lgt=lgt)
        for k in range(0,len(colsi)):
            colsi[k]=matplotlib.colors.colorConverter.to_rgb(colsi[k])
        colsi = np.array(colsi)
        xi = np.linspace(0.,1.,colsi.shape[0])

        # to interpolate between
        colsj = np.ndarray([N,3],dtype=float)
        xj = np.linspace(0.,1.,N)

        for k in range(0,3):
            f1=interp1d(xi,colsi[:,k],
                        bounds_error=False)
            colsj[:,k]=f1(xj)
        
        # collect for output
        cols = [tuple(colsj[k,:]) for k in range(0,N)]

        
    return cols
    

def boxpolygon(xlm,ylm):
    """
    :param     xlm:  x-limits
    :param     ylm:  y-limits
    """

    # to indices
    xlm = np.atleast_1d(xlm).astype(float)
    ylm = np.atleast_1d(ylm).astype(float)

    # vertices
    vt = np.ndarray([5,2],dtype=float)
    vt[:,0] = xlm[np.array([0,1,1,0,0])]
    vt[:,1] = ylm[np.array([0,0,1,1,0])]

    # create polygon
    ply = Polygon(vt)

    return ply


def delticklabels(pm,axs='both'):
    """
    :param   pm: grid of plots to delete labels for
                   first dimension is x
    :param  axs: labels to delete 'x','y', or 'both' (default)
    """

    szx = pm.shape[1]
    szy = pm.shape[0]

    if axs is 'both':
        delticklabels(pm,'x')
        delticklabels(pm,'y')
    elif axs is 'y':
        for ky in range(0,szy):
            for kx in range(1,szx):
                pm[ky,kx].set_yticklabels([])
    elif axs is 'x':
        for ky in range(0,szy-1):
            for kx in range(0,szx):
                pm[ky,kx].set_xticklabels([])

    return


def cornerlabels(p,loc='ll',fontsize='medium',scl=None):
    """
    :param        p:  list of plots
    :param      loc:  location (default: 'll'--lower left)
    :param fontsize:  font size (default: 'medium')
    :param      scl:  distance from edge
    """
    
    # set grid of values
    lbl='abcdefghijklmnopqrstuvwxyz'

    if scl is None:
        if loc[0]=='o':
            scl = .1
        else:
            scl = 0.95

    if loc is 'll':
        x,y=scl,scl
        hl = 'left'
        vl = 'bottom'
    elif loc is 'ul':
        x,y=scl,1.-scl
        hl = 'left'
        vl = 'top'
    elif loc is 'ur':
        x,y=1.-scl,1.-scl
        hl = 'right'
        vl = 'top'
    elif loc is 'lr':
        x,y=1.-scl,scl
        hl = 'right'
        vl = 'bottom'
    if loc is 'oll':
        x,y=-scl,-scl
        hl = 'left'
        vl = 'bottom'
    elif loc is 'oul':
        x,y=-scl,1.+scl
        hl = 'left'
        vl = 'top'
    elif loc is 'our':
        x,y=1.+scl,1.+scl
        hl = 'right'
        vl = 'top'
    elif loc is 'olr':
        x,y=1.+scl,-scl
        hl = 'right'
        vl = 'bottom'



   
    for k in range(0,len(p)):
        # correct label
        kk = k % 26
        jj = (k / 26) + 1
        ll = lbl[kk]*jj

        # plot
        p[k].text(x,y,ll,transform=p[k].transAxes,
                  fontsize=fontsize,alpha=1.,backgroundcolor='w',
                  horizontalalignment=hl,verticalalignment=vl,
                  bbox=dict(edgecolor='black',facecolor='w'))


def timelabel(tlen,nsig=2,wdash=True):
    """
    :param   tlen:   time length in second
    :param   nsig:   number of significant figures
    :param  wdash:   add a dash, as for 10-minute
    :return  tlbl:   label for this time
    """

    tlen = np.atleast_1d(tlen)

    # which value to use
    lmins = np.array([1.,60.,3600.,86400.])
    ix = np.searchsorted(lmins,tlen,'left')
    ix = np.maximum(ix-1,0)

    # divide by the relevant scaling
    vl = np.divide(tlen,lmins[ix])

    # round
    vl = general.roundsigfigs(vl,nsig)
    vl = np.atleast_1d(vl)
    
    # labels
    lbls = np.array(['seconds','minutes','hours','days'])
    lbls = lbls[ix]
    lbls = np.atleast_1d(lbls)

    tlbl = []
    for k in range(0,len(vl)):
        vli = vl[k]
        lbli = lbls[k]

        if vli % 1==0:
            vli=int(vli)

        if vli==1 or wdash:
            lbli=lbli[0:-1]

        if wdash:
            lbli='-'+lbli
        else:
            lbli=' '+lbli
        
        tlbl.append(str(vli)+lbli)

    return tlbl
