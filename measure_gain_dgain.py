"""
Compute GAIN, DGAIN parameters using
folded-stitched-calibrator scans

GAIN, DGAIN computed using coherence products.
Made for circular feeds


Take folded-stitched-calibrator scan,
run :make_np.py: on it,
run this file on it
"""
import os
import warnings

import numpy as np

from circ_pacv import read_pkl, MyPACV

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('measure_gain_dgain', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('pkl', help='Pickle file (output of make_np.py)',)
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-O','--outdir', help='Output directory', default='./', dest='odir')
    return ag.parse_args ()


if __name__ == "__main__":
    args = get_args ()
    ###################################################
    ### prepare files/filenames
    ###################################################
    FILE_UNCAL  = args.pkl
    base,_      = os.path.splitext ( os.path.basename ( FILE_UNCAL ) )
    solfile     = os.path.join ( args.odir, base + ".gdg_sol.npz" )
    ### all this goes into npz
    RET         = dict()
    RET['sol_from'] = FILE_UNCAL
    ###################################################
    ### read calibrator file
    ###################################################
    ## read
    pkg, freq, sc    = read_pkl ( FILE_UNCAL )
    nchan       = freq.size
    ## ON phase solve
    ## ON-phase is more than 60% of the maximum
    pp          = sc[0].mean(0)
    mask        = pp >= (0.60 * pp.max())
    ff          = sc[...,mask].mean(-1) - sc[...,~mask].mean(-1)
    #######################
    ## in case stokes-i (ff[0]) is negative, flag it.
    ## it should not be expected but if the calibrator scan is that bad
    ## then yea
    lz                     = ff[0] <= 0.0
    if np.any (lz):
        warnings.warn (f" ON-OFF Stokes-I is negative, this should not be", RuntimeWarning)
        freq.mask[lz]      = True
        sc.mask[:,lz,:]    = True
        ff.mask[...,lz]    = True
    #######################
    off_std     = sc[...,~mask].std(-1)
    #######################
    feed        = 'CIRC'
    if pkg['basis'] != 'Circular':
        raise RuntimeError (" Basis not supported =",pkg['basis'])
    ##################################################
    ### perform fitting
    ### we are borrowing code we wrote for :make_pacv.py: here
    ### we are simply going to use GAIN, DGAIN, ERRORS resp. here
    ###################################################
    caler          = MyPACV ( feed, freq, ff, off_std, 0.0, 0.0 )
    RET['gain']    = caler.gain 
    RET['gainerr'] = caler.gainerr
    RET['dgain']   = caler.dgain
    RET['dgainerr'] = caler.dgainerr
    RET.update ( pkg )
    RET['freq']    = freq
    RET['freq_mask'] = freq.mask
    ###################################################
    ## writing solution
    ###################################################
    np.savez ( solfile, **RET )
