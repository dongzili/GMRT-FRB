"""
saves output from read_prepare

so that rmsynth can be run
"""

import os
import sys

import numpy as np

from read_prepare import read_prepare_tscrunch 

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('save_rmsynth', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-f','--fscrunch', default=4, type=int, help='Frequency downsample', dest='fs')
    add ('pkg', help="package file output by make_pkg")
    add ('-n','--no-subtract', help='do not subtract off', action='store_true', dest='nosub')
    add ('-O','--outdir', help='Output directory', default='./', dest='odir')
    ##
    return ag.parse_args ()

if __name__ == "__main__":
    args = get_args ()
    ###
    arg = args.pkg
    ###
    bn  = os.path.basename ( arg )
    ###
    ofile = os.path.join ( args.odir, bn + ".synth" )

    freqs, i, q, u, v, ei, eq, eu, ev = read_prepare_tscrunch ( arg, args.fs, args.nosub)

    i = np.array ( i[...,0] )
    q = np.array ( q[...,0] )
    u = np.array ( u[...,0] )
    v = np.array ( v[...,0] )

    cat  = np.vstack ( (freqs, i, q, u, ei, eq, eu) ).T
    ## 7 columns
    ## freq, I Q U, errors I Q U

    np.savetxt ( ofile, cat )





