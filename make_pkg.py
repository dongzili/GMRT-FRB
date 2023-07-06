#!/usr/bin/python2.7
"""

makes a json package so that psrchive dependency can be removed

"""
from __future__ import print_function

import os
import sys
import json

import numpy as np

import psrchive

# from skimage.measure import block_reduce

def split_extension ( f ):
    r,_ = os.path.splitext (f)
    return r

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('make_pkg', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-O','--outdir', help='Output directory', dest='odir', default=None)
    add ('-j','--json', default=None, help="JSON file containing tstart,tstop,fstart,fstop", dest='json')
    add ('file', help="archive file")
    add ('-k','--keep-baseline', help='Keep baseline', action='store_true', dest='baseline')
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    ##
    return ag.parse_args ()

jl   = lambda x : [float(ix) for ix in x]

def read_ar (fname, remove_baseline=False):
    """
    reads
    """
    ff    = psrchive.Archive_load (fname)
    ff.convert_state ('Stokes')
    if remove_baseline:
        ff.remove_baseline ()
    ff.dedisperse ()
    ###
    nbin  = ff.get_nbin()
    nchan = ff.get_nchan()
    # dur   = ff.get_first_Integration().get_duration()
    dur   = ff.get_first_Integration().get_folding_period ()
    fcen  = ff.get_centre_frequency ()
    fbw   = ff.get_bandwidth ()
    fchan = fbw / nchan
    tsamp = dur / nbin
    ###
    start_time   = ff.start_time ().in_days ()
    end_time     = ff.end_time ().in_days ()
    mid_time     = 0.5 * ( start_time + end_time )
    ###
    src          = ff.get_source ()
    rmc          = ff.get_faraday_corrected ()
    pcal         = ff.get_poln_calibrated ()
    ###
    data  = ff.get_data ()
    #### making data and wts compatible
    ww = np.array (ff.get_weights ().squeeze(), dtype=bool)
    wts   = np.ones (data.shape, dtype=bool)
    wts[:,:,ww,:] = False
    mata  = np.ma.array (data, mask=wts, fill_value=np.nan)
    dd    = dict (nbin=nbin, nchan=nchan, dur=dur, fcen=fcen, fbw=fbw, 
        data=data, 
        # wts = wts, 
        wts = ww,
        rm_correction = rmc,
        polcal = pcal,
        # wts = jl ( ww ),
        # data_shape = jl ( data.shape ),
        # data_ravel = jl ( data.ravel () ),
        tsamp=tsamp, src=src, obstime=mid_time)
    return dd

if __name__ == "__main__":
    args    = get_args ()
    bn      = os.path.basename ( args.file )
    if args.odir:
        outfile = os.path.join ( args.odir, bn + ".pkg" )
    else:
        outfile =  args.file + ".pkg"
    ## read file and ranges
    pkg     = read_ar ( args.file, remove_baseline = not args.baseline )

    ##
    RET     = dict ()
    RET.update (pkg)
    RET['filename']     = bn

    if args.json:
        with open (args.json, 'rb') as f:
            ran = json.load (f)
    else:
        with open (args.file+".json", 'r') as f:
        # with open (args.file[:-5]+".calibP.Czap.json", 'r') as f:
            ran = json.load (f)

    RET.update ( ran )
    del RET['file']
    ## save
    # with open (outfile, 'w') as f:
        # json.dump ( RET, f )
    np.savez_compressed ( outfile, **RET  )
    # with open (outfile, 'w') as f:
        # json.dump ( RET, f )

