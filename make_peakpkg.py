#!/usr/bin/python2.7
"""

make a npz package to remove psrchive dependency

designed for pulsars which takes full band at pulsar peak phase.

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
    ag   = agp.ArgumentParser ('make_peakpkg', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-O','--outdir', help='Output directory', dest='odir', default=None)
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
    ####
    ipp   = mata[0,0].mean(0) # bin
    ibin  = ipp.argmax ()
    fsl   = mata[0,...,ibin]
    jbin  = ipp.argmin()
    __offslice = slice(max(jbin-5,0),min(jbin+5,nbin))
    bsl   = mata[0,...,jbin]
    esl   = mata[0,...,__offslice].std(-1)
    ### 
    ## fsl will be (4,2048)
    # (subint, pol, chan, bin)
    dd    = dict (
        nbin=nbin, nchan=nchan, dur=dur, fcen=fcen, fbw=fbw, 
        # data=data, 
        max_slice = np.array(fsl),
        mask_max  = np.array(fsl.mask),
        min_slice = np.array(bsl),
        mask_min  = np.array(bsl.mask),
        std_slice = np.array (esl),
        mask_std  = np.array(esl.mask),
        # wts = wts, 
        # wts = ww,
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
        outfile = os.path.join ( args.odir, bn + ".peakpkg" )
    else:
        outfile =  args.file + ".peakpkg"
    ## read file and ranges
    pkg     = read_ar ( args.file, remove_baseline = not args.baseline )

    ##
    RET     = dict ()
    RET.update (pkg)
    RET['filename']     = bn

    np.savez_compressed ( outfile, **RET  )


