import os,glob
import general
import numpy as np
import obspy
import time,datetime

def defcalib(stn='B004'):
    """
    :param     stn:  station
    :return  calib:  calibration name
    """
    
    # preferred calibration
    clb=['B003','B004','B014','B024','B027','B028',
         'B030','B031','B032','B033','B035','B036',
         'B039','B040','B045','B057','B058','B065',
         'B067','B073','B075','B076','B078','B079',
         'B081','B082','B084','B086','B087','B088',
         'B093','B201','B202','B023','B206','B901',
         'B916','B918','B921','B926','B934','B935',
         'B941','B944']
    clb=dict.fromkeys(clb,'HodgkinsonPBO')
    calib=clb.get(stn,'Hawthorne2')

    return calib
    

def readcalib(stn,ctyp='HodgkinsonPBO'):
    """
    read calibration matrix for the specified station
    :param        stn: station
    :param       ctyp: calibration type
    :return      cmat: calibration matrix
    :return startcomp: starting components
    :return   endcomp: ending components
    """

    # file to read
    fdir = os.environ['STRAINPROC']
    fname=os.path.join(fdir,'CALIBMAT',ctyp,stn)


    fl=open(fname,'r')

    # get the components mapped
    startcomp=fl.readline().split(',')
    endcomp=fl.readline().split(',')
    Ne=len(endcomp)
    Ns=len(startcomp)

    # strip the ends
    for k in range(0,len(startcomp)):
        startcomp[k]=startcomp[k].strip()
    for k in range(0,len(endcomp)):
        endcomp[k]=endcomp[k].strip()


    for line in fl:
        # try each line
        vl=line.split()

        if vl:
            # if it's this station
            if vl[0]==stn:
                cmat=vl[1:]
                cmat = np.reshape(np.array(cmat),[Ne,Ns])
    fl.close()
    cmat=cmat.astype(float)

    return cmat,startcomp,endcomp

def readpbotrend(stn,st=None):
    """
    :param    stn:  station
    :param     st:  waveforms---for reference date if not given
    :return     X:  fit coefficients
    :return  fpar:  fit parameters
    """

    # file with info
    fname=os.path.join(os.environ['DATA'],'STRAINPBO',stn,
                       stn+'.README.txt')

    # read beginning
    fl=open(fname,'r')
    line=fl.readline()
    while not 'Borehole Trend Models' in line:
        line=fl.readline()

    # skip header
    fl.readline()
    
    # initialize
    X={}
    fpar={'fitconstant':1,'fitlinear':1,'fitexp':1}
    fpar['chfit']=['G0','G1','G2','G3']
    fpar['expdec']=dict.fromkeys(fpar['chfit'],None)

    for k in range(0,4):
        line=fl.readline()
        vls=line.split()

        # set channel name and initialize
        ch,Xi=channame(vls[0]),{}

        # constant
        Xi['constant']=np.array(float(vls[1]))*1e-9

        # linear trend
        Xi['linear']=np.array(float(vls[4]))*1e-9

        # exponential amplitudes
        Xi['exp']=np.array([float(vls[2]),float(vls[5])])*1e-9

        # exponential decay constants
        dcy=np.array([float(vls[3]),float(vls[6])])
        dcy[dcy==0]=-1.
        fpar['expdec'][ch]=np.divide(1,dcy)

        # for collection
        X[ch] = Xi

    # reference time
    line=fl.readline()
    tm=line.split()
    tm=tm[-1].strip()

    # close
    fl.close()


    try:
        tm=obspy.UTCDateTime(tm)
    except:
        # use install date if reference time isn't given
        fl=open(fname,'r')
        line=fl.readline()
        while not 'Install Date' in line:
            line=fl.readline()
        fl.close()
        line = line.split()
        tm=obspy.UTCDateTime(line[-1])
        #tm=st[0].stats.starttime-86400.

    fpar['exptref']=tm
    fpar['starttime']=tm
    fpar['endtime']=tm


    # return parameters
    return X,fpar

def qualtexttosac(stn=None):
    """
    read quality data from text and write to a sac file
    :param    stn: station name or names 
              (default: read from $DATA/STRAINPBO/statlistbest)
    """

    if stn is None:
        stn = os.path.join(os.environ['DATA'],'STRAINPBO','statlistbest')
        stn = np.loadtxt(stn,dtype=str)

    if isinstance(stn,str):
        print('Rewriting strain quality data for '+stn)

        # read the data
        st = readqualtext(stn)
    
        # directory to write to
        fdir = os.path.join(os.environ['DATA'],'STRAINPBO','QUALSAC')

        for tr in st:
            fname='strainquality-'+tr.stats.station+'-'+tr.stats.channel+'.SAC'

            # fill dropouts 
            if isinstance(tr.data,np.ma.masked_array):
                if np.sum(tr.data.mask):
                    tr.data[tr.data.mask]=4
                tr.data=tr.data.data

            # write to file
            tr.write(filename=os.path.join(fdir,fname),format='SAC')

    else:
        # for each station
        for stni in stn:
            qualtexttosac(stni)
    

def readqualtext(stn,cmps=['gauge0','gauge1','gauge2','gauge3']):
    """
    :param    stn: station name
    :param   cmps: components to read 
                    (default: ['gauge0','gauge1','gauge2','gauge3'])
    :return    st: waveforms with 
                     1: good, 2: bad, 3: missing, 4: interpolated
    """

    # relevant directory
    fdir = os.path.join(os.environ['DATA'],'STRAINPBO',stn)
    drs = glob.glob(os.path.join(fdir,stn+'*.bsm.level2'))

    # all the channels
    chns = np.array([])

    # all the streams
    st = obspy.Stream()

    # to map to quality
    ps = np.array(['b','g','m','i'])
    nmap = np.array([2,1,3,4])

    for cm in cmps:
        for dr in drs:
            fls=glob.glob(os.path.join(fdir,dr,stn+'*.xml.'+cm+'.txt'))
            for fl in fls:
                print(fl)
                try:
                    # first try a simple load
                    data=np.loadtxt(fl,dtype=str,skiprows=1)

                except:
                    # but if there are irregularities
                    fli = open(fl,'r')
                    trash = fli.readline()
                    
                    # read but skip lines with the wrong number of columns
                    data = [line.split() for line in fli]
                    data = [vl for vl in data if len(vl)==14]
                    data = np.array(data)
                    data = np.atleast_2d(data)

                    fli.close()

                # data quality
                qual = data[:,6]
                qual = np.searchsorted(ps,qual)
                qual = nmap[qual].astype(float)
                
                # timing
                tms=np.array([obspy.UTCDateTime(tm) for tm in data[:,1]])
                tdf = tms - tms[0]
                tms=np.array([obspy.UTCDateTime(data[0,1]),
                              obspy.UTCDateTime(data[-1,1])])
                
                # channel here
                ch = data[0,0]
                
                tr = st.select(channel=ch)
                if tr:
                    # if this waveform already exists
                    tr = tr[0]
                    dtim = tr.stats.delta
                else:
                    # if not
                    tr = obspy.Trace()
                    tr.stats.station = stn
                    tr.stats.network = 'PB'
                    tr.stats.channel = ch
                    
                    # time spacing
                    dtim=np.median(np.diff(tdf))
                    dtim=general.roundsigfigs(dtim,6)
                    tr.stats.delta = dtim
                    tr.stats.starttime=tms[0]
                    tr.data = qual[0:1]
                    
                    st.append(tr)

                # add times
                nadd = int(np.round((tms[-1]-tr.stats.endtime)/dtim))
                tint = (tr.stats.endtime-tms[0])+np.arange(1.,nadd+1.)*dtim
                ii = general.closest(tdf,tint)
                qual = np.ma.masked_array(qual[ii],mask=np.abs(tdf[ii]-tint)>dtim)

                tr.data = np.append(tr.data,qual)
            

    # change the names to match convention
    for tr in st:
        tr.stats.channel=channame(tr.stats.channel)
        tr.stats.channel='DQ'+tr.stats.channel

    return st


def gagetolinstrain(st):
    """
    convert the gage observations to linearize strain
    :param     st: waveforms
                    modified in place
    """

    for tr in st:
        # gap and diameter
        statprop=pbometadata(tr.stats.station)
        gp=statprop['GAP(m)']
        diam=statprop['DIAM(m)']

        # reference value
        tri = 'L'+tr.stats.channel[-1]+'(cnts)'
        tri = statprop.get(tri,None)

        # if it wasn't given
        if tri is None:
            tri=tr.copy().trim(tr.stats.starttime,
                               tr.stats.starttime+300.*86400.)
            tri=tri.data
            tri=tri[tri!=999999]
            tri=tri[1:500]
            tri=np.mean(tri)
        
        # shift
        ix=tr.data!=999999
        tr.data[ix]=(np.divide(tr.data[ix]/1e+8,1-(tr.data[ix]/1e+8)) - 
                     (tri/1.e+8)/(1.-(tri/1.e+8)))*gp/diam


def identpbofiles(stn,chn=None,usedb=False,tlm=None):
    """
    :param      stn:  station
    :param      chn:  list of channels
    :param    usedb:  use the pbo database (default: True)
    :param      tlm:    time limits to allow
    :return     fls:  list of relevant files
    """

    # channels for data
    if chn is None:
        chn=['RS1','RS2','RS3','RS4']

    # needs to be a list
    if isinstance(chn,str):
        chn=[chn]

    if usedb:
        import pisces
        import databseis
        from piscestables import Waveform

        # get basic values
        session = databseis.opendatabase('pbostrain')
        q = session.query(Waveform)
        q = q.filter(Waveform.sta==stn)
        q = q.filter(Waveform.net=='PB')

        if tlm:
            q = q.filter(Waveform.time<=tlm[1])
            q = q.filter(Waveform.endtime>=tlm[0])

        fls = []
        for ch in chn:
            qq = q.filter(Waveform.chan==ch)
            flsi = [os.path.join(wv.dir,wv.dfile) for
                    wv in qq]
            fls = fls+flsi

        session.close()
    else:
        # directory with files
        fdir=os.path.join(os.environ['DATA'],'STRAINPBO','SACDATA')

        fls=[]
        for ch in chn:
            fls=fls+glob.glob(os.path.join(fdir,'*PB.'+stn+'*'+ch+'*.SAC'))


    return fls

def readpbogagedata(stn,pfx='R',fls=None,chqual=True,tlm=None):
    """
    read the gage data in the specified directory
    :param    stn:    station
    :param    pfx:    prefix for sampling rate (default: 'R': 10-min data)
    :param    fls:    list of files (default: all with file name)
    :param chqual:    check the quality control and delete problematic times
                        (default: True)
    :param    tlm:    time limits to allow
    :return    st:    waveforms
    """

    # channels for data
    chn=['S1','S2','S3','S4']
    for k in range(0,len(chn)):
        chn[k]=pfx+chn[k]


    # identify available files 
    if fls is None:
        fls=identpbofiles(stn,chn=chn,usedb=True,tlm=tlm)



    # initialize waveforms
    st=obspy.Stream()

    # read files
    for fnm in fls:
        # read
        sti = obspy.read(fnm)
        st.append(sti[0])

    # add a mask for problematic data
    for tr in st:
        idel=tr.data==999999
        tr.data=np.ma.array(data=tr.data,mask=idel,
                            fill_value=None)

    # delete known outliers
    delints(st,toutliers(stn))

    # merge so there's one waveform per channel
    st=st.merge()

    # convert the data to linearized strain
    gagetolinstrain(st)

    if chqual:
        flsq=None
        # identify available files 
        if flsq is None:
            chq = ['DQG0','DQG1','DQG2','DQG3']
            flsq=identpbofiles(stn,chn=chq,usedb=True)        

        # initialize waveforms
        stq=obspy.Stream()

        # read files
        for fnm in flsq:
            stq=stq+obspy.read(fnm)

        # channels for quality indicators
        for tr in st:
            lb=str(int(tr.stats.channel[-1])-1)
            chqi=['DQG'+lb]
            stqi=obspy.Stream()
            for ch in chqi:
                stqi=stqi+stq.select(channel=ch)
                stqi[-1].stats.channel=chqi[0]
            stqi=stqi.merge()
            
            # change to common time sampling
            dtim=tr.stats.delta
            dtimm=np.maximum(dtim,stqi[0].stats.delta)
            tmin=tr.stats.starttime
            stqi=stqi.trim(starttime=tr.stats.starttime-2*dtimm,
                           endtime=tr.stats.endtime+2*dtimm,
                           pad=True,fill_value=0.)
            stqi.interpolate(1./dtim,starttime=tmin,method='linear')

            # mask anything that's not 1
            imsk=np.abs(stqi[0].data[0:tr.stats.npts]-1.)>0.01
            imsk=np.logical_or(imsk,tr.data.mask)
            tr.data=np.ma.array(data=tr.data.data,mask=imsk,
                                fill_value=None)

    # trim out the bits at the end without data
    tmax=np.min([tr.stats.starttime for tr in st])
    tmin=np.max([tr.stats.endtime for tr in st])
    for tr in st:
        ii,=np.where(st[0].data.mask)
        if ii.any():
            tmin=np.minimum(tmin,tr.stats.starttime+
                            ii[0]*tr.stats.delta)
            tmax=np.maximum(tmax,tr.stats.starttime+
                            ii[-1]*tr.stats.delta)
    st.trim(starttime=tmin,endtime=tmax)
        
    return st

def readpboatmdata(stn,fls=None):
    """
    read the gage data in the specified directory
    :param    stn:    station
    :param    fls:    list of files (default: all with file name)
    :return    st:    waveforms
    """

    # channels for data
    chn = ['RDO']

    # identify available files 
    if fls is None:
        fls=identpbofiles(stn,chn=['RDO'],usedb=True)

    # identify available files 
    if fls is None:
        # directory with files
        fdir=os.path.join(os.environ['DATA'],'STRAINPBO','SACDATA')

        fls=[]
        for ch in chn:
            fls=fls+glob.glob(os.path.join(fdir,'*PB.'+stn+'*'+ch+'*.SAC'))

    # initialize waveforms
    st=obspy.Stream()

    # read files
    for fnm in fls:
        # read
        sti = obspy.read(fnm)
        st.append(sti[0])

    # add a mask for problematic data
    for tr in st:
        idel=tr.data==999999
        tr.data=np.ma.array(data=tr.data,mask=idel,
                            fill_value=None)

    # delete known outliers
    delints(st,toutliers(stn+'-atm'))

    # merge so there's one waveform
    st=st.merge()

    # trash data at edges
    st=st.split().merge()

    return st

def read(stn,pfx='R',calib='default',fls=None,flsa=None,chqual=True,
         tlm=None):
    """
    :param    stn:    station
    :param    pfx:    prefix for sampling rate (default: 'R')
    :param  calib:    name of directory with calibration file
                         (default: 'HodgkinsonPBO')
    :param    fls:    list of files (default: all with file name)
    :param   flsa:    list of atm files (default: all with file name)
    :param chqual:    check the quality controlled data
    :param    tlm:    time limit of data to get
    :return    st:    waveforms
    """

    # read the gauge data
    st=readpbogagedata(stn,pfx=pfx,fls=fls,chqual=chqual,tlm=tlm)

    # just common times
    tmin=st[0].stats.starttime
    tmax=st[0].stats.endtime
    dtim=st[0].stats.delta
    for tr in st:
        tmin=max(tmin,tr.stats.starttime)
        tmax=min(tmax,tr.stats.endtime)

    # read the barometric data
    sta=readpboatmdata(stn,fls=flsa)

    # high-pass filter atmospheric data?
    sta=sta.split().detrend()
    #sta=dtfilt(sta.merge(),[1/30./86400.,float('inf')],3.*86400.)
    sta=sta.merge()

    # change to common time sampling
    sta.trim(starttime=tmin-3*dtim,endtime=tmax+3*dtim,pad=True)
    data=sta[0].data.data
    data[sta[0].data.mask]=999999
    sta[0].data=data
    sta.interpolate(1./dtim,starttime=tmin,method='nearest')
    sta[0].data=np.ma.array(data=sta[0].data,mask=sta[0].data==999999)

    # combine data
    st = st + sta
    st.trim(starttime=tmin,endtime=tmax)

    # rename channels with default names
    for tr in st:
        tr.stats.channel=channame(tr.stats.channel)
        tr.stats.sac.kcmpnm=tr.stats.channel

    # and calibrate
    if calib is 'default':
        calib=defcalib(stn)
    if calib:
        st=chcomp(st,calib=calib)
         
    return st

def channame(chorig):
    """
    :param     chorig: original channel name
    :return    chname: simplified channel name
    """

    dct={}

    dct['RS1']='G0'
    dct['RS2']='G1'
    dct['RS3']='G2'
    dct['RS4']='G3'

    dct['CH0']='G0'
    dct['CH1']='G1'
    dct['CH2']='G2'
    dct['CH3']='G3'

    dct['LS1']='G0'
    dct['LS2']='G1'
    dct['LS3']='G2'
    dct['LS4']='G3'

    dct['gauge0']='G0'
    dct['gauge1']='G1'
    dct['gauge2']='G2'
    dct['gauge3']='G3'

    dct['Eee+Enn']='E+N'
    dct['Eee-Enn']='E-N'
    dct['2Ene']='2EN'

    dct['gamma1']='E-N'
    dct['gamma2']='2EN'

    # get the relevant value
    chname=dct.get(chorig,chorig)

    return chname


def chcomp(st,calib='HodgkinsonPBO'):
    """
    change components
    :param      st: original waveforms
    :param   calib:  calibration type (default: 'HodgkinsonPBO')
    """

    # calibration matrix
    cmat,startcomp,endcomp=readcalib(st[0].stats.station,calib)

    # project
    sta=obspy.Stream()

    # initialize trace
    tr = st[0].copy()

    # change  component name
    for tri in st:
        tri.stats.channel=channame(tri.stats.channel)

    # go through each new component
    for k1 in range(0,len(endcomp)):
        # set to zero
        tr.data[:]=0
        tr.stats.channel=endcomp[k1]
        
        # and each old component
        for k2 in range(0,len(startcomp)):
            tri=st.select(channel=startcomp[k2])[0]
            tr.data=tr.data+cmat[k1,k2]*tri.data

        # add to set
        sta.append(tr.copy())

    # add atmospheric
    st=sta+st.select(channel='RDO')

    return st


def toutliers(stn):
    """
    return times with manually identified problems in the data
    :param    stn:   station of interest
    :return  tdel:   2 x ? array of intervals to remove
    """

    # file to read
    fname=os.path.join(os.environ['STRAINPROC'],'OUTLIERS')
    fname=os.path.join(fname,stn)
    
    # initialize list
    tdel=[]

    # go through files
    if os.path.isfile(fname):
        fl = open(fname,'r')
        for line in fl:
            vls=line.split(',')
            tdel.append(obspy.UTCDateTime(vls[0]))
            tdel.append(obspy.UTCDateTime(vls[1]))
        fl.close()
        
    # shape
    tdel=np.array(tdel)
    tdel=tdel.reshape(2,len(tdel)/2)

    return tdel


def pbostationlist():
    """
    read  metadata from file
    :return  stns:    list of stations
    """

    # file name
    fname=os.path.join(os.environ['STRAINPROC'],'METADATA')
    fname=os.path.join(fname,'bsm_metadata.txt')

    # read 
    fl=open(fname,'r')
    
    # initialize
    stns = []
    
    for line in fl:
        # values
        vls=line.replace('->',' ')
        vls=vls.split()
        stns.append(vls[0])

    return stns


def pbometadata(stn):
    """
    read  metadata from file
    :param    stn:    station to read
    :return  avls:    dictionary of instrument properties
    """

    # file name
    fname=os.path.join(os.environ['STRAINPROC'],'METADATA')
    fname=os.path.join(fname,'bsm_metadata.txt')

    # read 
    fl=open(fname,'r')
    
    # header
    hdr=fl.readline()
    hdr=hdr.split()

    # to change to relevant values
    avls=dict.fromkeys(hdr)

    # to change to float
    hfl=['BSM_Depth(m)','CH0(EofN)','ELEV(m)','GAP(m)',
         'L0(cnts)','L1(cnts)','L2(cnts)','L3(cnts)',
         'LAT','LONG','SEISMOMETER_Depth(m)']
    hfl2=['PORE_DEPTH(m)']

    for line in fl:
        # values
        vls=line.replace('->',' ')
        vls=vls.split()

        # if it's the right station
        if vls[0]==stn:
            avls=dict(zip(hdr,vls))
    
            # set to float
            for ky in hfl:
                if avls[ky]=='Unknown':
                    avls[ky] = float('nan')
                else:
                    avls[ky]=float(avls[ky])

            for ky in hfl2:
                if avls[ky]!='NA':
                    avls[ky]=float(avls[ky])

    # all the diameters are the same
    avls['DIAM(m)']=0.087

    fl.close()

    return avls


def delints(tr,tdel):
    """
    go through and mask values in traces or streams tr within range tdel
    :param    tr:   trace or stream
    :param  tdel:   a list or array of intervals to delete, 
                alternating start and end times
    nothing returned---done in place
    """

    if tdel is None:
        tdel = []

    if isinstance(tr,obspy.Stream):
        # go through the traces
        for tri in tr:
            delints(tri,tdel)
    else:
        if len(tdel):
            # mask the value if not given
            if ~isinstance(tr.data,np.ma.masked_array):
                tr.data=np.ma.array(tr.data)
                
            # subtract from start time
            tdel = tdel-tr.stats.starttime
            tim = tr.times()
            
            tdel=np.atleast_2d(tdel)
            if tdel.size:
                tdel=tdel.reshape([tdel.size/2,2])
            
            # go through and find times
            for k in range(0,tdel.size/2):
                ih=np.logical_and(tim>=tdel[k,0],tim<=tdel[k,1])
                tr.data.mask=np.logical_or(tr.data.mask,ih)

def readcorrstrain(stn,chn=None,apnd='-100'):
    """
    write the data to sac files
    :param   st:  waveforms
    :param  chn:  channels (default: ['E+N-na','E-N-na','2EN-na'])
    :param apnd:  label to append to file name (default: '-100')
    :return  st:  the strain time series
    """

    if chn is None:
        chn = ['E+N-na','E-N-na','2EN-na']
    if isinstance(chn,str):
        chn = [chn]

    # directory to read from
    fdir = os.path.join(os.environ['STRAINPROC'],'PROCESSED')

    # initialize the waveforms
    st = obspy.Stream()

    for ch in chn:
        # file name
        nm = 'strain-'+stn+'-'+ch+'-corr'+apnd+'.SAC'
        nm = os.path.join(fdir,nm)

        # read
        st = st+obspy.read(nm)

    for tr in st:
        tr.data = np.ma.array(tr.data,mask=tr.data==999999)
        
    return st


def writestrain(st,apnd):
    """
    write the data to sac files
    fills gaps with 999999
    :param   st:  waveforms
    :param apnd:  label to append to file name
    """

    # directory to write to
    fdir = os.path.join(os.environ['STRAINPROC'],'PROCESSED')

    # create diretory if it doesn't exist
    if not os.path.exists(fdir):
        print('Creating directory for data: '+fdir)
        os.makedirs(fdir)

    for tr in st:
        # file name
        nm = 'strain-'+tr.stats.station+'-'+tr.stats.channel+apnd+'.SAC'
        nm = os.path.join(fdir,nm)

        if isinstance(tr.data,np.ma.masked_array):
            tri=tr.copy()
            msk=tri.data.mask
            tri.data=tri.data.data
            tri.data[msk]=999999
        else:
            tri=tr

        # make sure the channel and station are copied correctly
        tri.stats.sac.kcmpnm=tri.stats.channel
        tri.stats.sac.knetwk=tri.stats.network
        tri.stats.sac.kstnm=tri.stats.station
        tri.stats.sac.delta = tri.stats.delta

        # write
        tri.write(nm,format='SAC')
        print('Wrote file '+nm)

    return


def localstat(loc='centcasc'):
    """
    :param     loc:  location---'centcasc','sanjuanbautista','parkfield'
    :return   stns:  a list of stations
    """
    
    if loc is 'sjb':
        loc = 'sanjuanbautista'
    elif loc is 'central cascadia':
        loc = 'centcasc'

    # file
    fdir = os.path.join(os.environ['STRAINPROC'],'METADATA')
    fname = os.path.join(fdir,loc+'stat')
    
    if os.path.exists(fname):
        # read
        stns = np.loadtxt(fname,dtype=str)
    else:
        print('No list of stations: file '+fname+' not found')

    return stns

