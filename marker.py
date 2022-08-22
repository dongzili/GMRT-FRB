
import os
import json

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import astropy.io.fits as aif

from skimage.measure import block_reduce

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('marker', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('file', help="calibrated pazi archive file")
    add ('-t','--tscrunch', help='Time scrunch', default=4, type=int, dest='ts')
    add ('-f','--fscrunch', help='Freq scrunch', default=16, type=int, dest='fs')
    return ag.parse_args ()

def dd_process (idd):
    """preprocesses filterbank"""
    nch, nbin = idd.shape
    ### remove per channel mean/std.dev
    odd    = np.float32 (idd)
    odd    -= np.mean(odd, 1).reshape ((nch, 1))
    sshape = np.std (odd,1).reshape ((nch, 1))
    odd    = np.divide (odd, sshape, out=np.zeros_like (odd), where=sshape != 0.)
    return odd


def read_ar (f):
    """reads the AR file without using psrchive """
    f       = aif.open (f)
    #### get SUBINT table
    names   = [fi.name for fi in f]
    idx     = None
    try:
        idx = names.index ('SUBINT')
    except:
        raise RuntimeError (" SUBINT table not found")
    #### get nchan, npol
    fh      = f[idx].header
    fd      = f[idx].data
    nchan   = fh['NCHAN']
    npol    = fh['NPOL']
    #### get scales, offsets, weights and data
    scl     = fd['DAT_SCL'].reshape ((1, npol, nchan, 1))
    offs    = fd['DAT_OFFS'].reshape ((1, npol, nchan, 1))
    wts     = fd['DAT_WTS']
    mask    = wts[0] == 0.
    #### get stokes I filterbank and apply mask
    dd      = (scl * fd['DATA']) + offs
    dd[...,mask,:] = np.nan
    #### #### coherence products still
    idd     = dd[0,0] + dd[0,1]
    return dd_process ( idd )


def get_ranges ( ret, fac ):
    """
    ret is list of size 2 where
    ret = [ [a,b], [c,d] ]
    two corners of rectangle

    first index is freq, second index is time
    """
    if len (ret) != 2:
        raise RuntimeError ("Two points not selected")

    A       = ret[0]
    B       = ret[1]

    m0      = min ( A[0], B[0] )
    m1      = min ( A[1], B[1] )

    M0      = max ( A[0], B[0] )
    M1      = max ( A[1], B[1] )

    trange  = ( m0, M0 )
    frange  = ( m1, M1 )

    saverange = dict ( 
        tstart = int (trange[0]*fac[1]), tstop = int (trange[1]*fac[1]) ,
        fstart = int (frange[0]*fac[0]), fstop = int (frange[1]*fac[0]) ,
    )
    return saverange

def on_press ( event ):
    """ on pressing mark key """
    global SPOINTS
    global ss
    if event.key == "m":
        ix, iy    = event.xdata, event.ydata
        SPOINTS.append ( (ix, iy) )
        ss.set_offsets ( SPOINTS )

    global fig
    fig.canvas.draw ()
    fig.canvas.flush_events ()


if __name__ == "__main__":
    ##
    args    = get_args ()
    FAC     = (args.fs, args.ts)
    ##
    dd      = read_ar ( args.file )
    dd      = block_reduce ( dd, FAC, func=np.nanmean )
    ##
    pp      = np.nanmean ( dd, 0 )
    ff      = np.nanmean ( dd, 1 )
    times   = np.arange (dd.shape[1])
    freqs   = np.arange (dd.shape[0])
    #############################
    SPOINTS   = []
    ####
    fig        = plt.figure ()

    fig.canvas.mpl_connect ('key_press_event', on_press)

    ax      = plt.subplot2grid ( (5,5), (1,0), rowspan=4, colspan=4, fig=fig )
    px      = plt.subplot2grid ( (5,5), (0,0), rowspan=1, colspan=4, fig=fig )
    bx      = plt.subplot2grid ( (5,5), (1,4), rowspan=4, colspan=1, fig=fig )


    ax.imshow ( dd, aspect='auto', interpolation='none', cmap='plasma', origin='lower' )
    px.step ( times, pp, where='mid', lw=1, color='blue' )

    bx.step ( ff, freqs, where='mid', lw=1, color='red' )

    ss      = ax.scatter ([], [], marker='s', color='black', s=60 )

    ax.get_shared_x_axes().join (ax, px)
    ax.get_shared_y_axes().join (ax, bx)

    ax.set_xlabel ('Time [units]')
    ax.set_ylabel ('Freq [units]')


    plt.show ()

    #####


    saverange = get_ranges ( SPOINTS, FAC )
    saverange['file'] = os.path.basename (args.file)

    with open (args.file + ".json", "w") as f:
        json.dump (saverange, f)
