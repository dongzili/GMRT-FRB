"""
saves output from read_prepare on all archives

so that one RM fitting can be run

"""

import os
import sys

import numpy as np

from read_prepare import read_prepare_tscrunch 
from scipy.ndimage import gaussian_filter1d

from collections import defaultdict

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('save_rmsynth', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-o','--outfile', help='Output file', required=True, dest='ofile')
    add ('-b','--smooth', default=4, type=float, help='Gaussian smoothing sigma', dest='bw')
    add ('-f','--fscrunch', default=8, type=int, help='Frequency downsample', dest='fs')
    add ('pkg', help="package file output by make_pkg", nargs='+')
    add ('-n','--no-subtract', help='do not subtract off', action='store_true', dest='nosub')
    ##
    return ag.parse_args ()

if __name__ == "__main__":
    args = get_args ()
    ###############################
    pkg = defaultdict(list)
    for arg in args.pkg:
        freqs, i, q, u, v, ei, eq, eu, ev = read_prepare_tscrunch ( arg, args.fs, args.nosub)

        i = np.array ( i[...,0] )
        q = np.array ( q[...,0] )
        u = np.array ( u[...,0] )
        v = np.array ( v[...,0] )

        ### smooth the Stokes
        if args.bw > 0:
            qfit  = gaussian_filter1d ( q, args.bw )
            ufit  = gaussian_filter1d ( u, args.bw )
        else:
            qfit  = q
            ufit  = u
        ###
        lfit = np.sqrt ( qfit**2 + ufit**2 )
        el   = np.sqrt ( eq**2 + eu**2 )
        ## get fractional
        qi  = q / lfit
        ui  = u / lfit

        qierr = np.sqrt ( eq**2 + eu**2 )
        uierr = np.sqrt ( eq**2 + eu**2 )
        ## same error?
        ## added in quad Stokes-Q,-U errors

        ## drop when either qi or ui > 1.
        ## it is something funky
        _m  = np.logical_or ( np.abs(qi) > 1., np.abs(ui) > 1. )

        pkg['freqs'].append ( freqs )
        pkg['qi'].append ( qi )
        pkg['ui'].append ( ui )
        pkg['qierr'].append ( qierr )
        pkg['uierr'].append ( uierr )
        pkg['mask'].append ( _m )

        # pkg['i'].append ( i )
        # pkg['q'].append ( q )
        # pkg['u'].append ( u )
    ###############################
    ukg = dict()
    for k,v in pkg.items():
        ukg[k] = np.concatenate ( v )
    ###############################
    np.savez ( args.ofile, **ukg )











