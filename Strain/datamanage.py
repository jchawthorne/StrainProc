import os,glob
import shutil
import numpy as np
import datarequests
import databseis
import obspy
import datetime
from piscestables import Waveform


def addqualdata(stn='*',datab='pbostrain'):
    """
    :param     stn:  station or stations to consider 
                      (default: '*'--all SAC files)
    :param   datab:  database to add to (default: 'pbostrain')
    """
    
    if isinstance(stn,str):
        stn = [stn]

    # open the relevant database
    session = databseis.opendatabase(datab)
    q = session.query(Waveform)

    # directory with data
    fdir = os.path.join(os.environ['STRAINPROC'],'QUALSAC')
    
    # create a temporary directory with copies of the data
    curdatai=os.environ['SDATA']
    curdata=datetime.datetime.now().strftime("%Y.%B.%d.%H.%M.%S.%f")
    curdata='curqualdata_' + curdata
    curdata=os.path.join(curdatai,curdata)
    os.makedirs(curdata)

    for stni in stn:
        # the relevant files
        fls = glob.glob(os.path.join(fdir,'*'+stni+'*.SAC'))
        
        for fl in fls:
            # for each file
            st = obspy.read(fl,headonly=True)
            
            # delete any relevant files already in the database
            qq = q.filter(Waveform.sta==st[0].stats.station)
            qq = qq.filter(Waveform.net==st[0].stats.network)
            qq = qq.filter(Waveform.chan==st[0].stats.channel)

            for wv in qq:
                # want to delete these files
                fname = os.path.join(wv.dir,wv.dfile)
                try:
                    os.remove(fname)
                except:
                    print('No file '+fname)
                
                # and the database entry
                session.delete(wv)

            # copy these files to the temporary directory
            head,tail = os.path.split(fl)
            shutil.copy(fl,os.path.join(curdata,tail))
            
    # commit and close
    session.commit()
    session.close()

    # add the files to the database
    databseis.allsort(datab=datab,fdirf=curdata,dlist=True)

    # delete the directory
    os.rmdir(curdata)

