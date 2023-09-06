"""

plots the package

"""
import os
import sys
import json

import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.colors as mc

from scipy.ndimage import gaussian_filter1d

from read_prepare import read_prepare_tscrunch, read_prepare_max
# from skimage.measure import block_reduce

def split_extension ( f ):
    r,_ = os.path.splitext (f)
    return r

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('vis_pkg', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-b','--smooth', default=4, type=float, help='Gaussian smoothing sigma', dest='bw')
    add ('-f','--fscrunch', default=8, type=int, help='Frequency downsample', dest='fs')
    add ('-s','--frange', default=None, nargs=2, type=int, help='Frequency range to keep', dest='frange')
    add ('-c','--choice', default='ts', choices=['ts','max'], help='what kind of visualization', dest='ch')
    add ('pkg', help="package file output by make_pkg")
    add ('-n','--no-subtract', help='do not subtract off', action='store_true', dest='nosub')
    add ('-i','--divide-i', help='Divide by I', action='store_true', dest='divi')
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-o','--ofile', help='Outplot file', dest='ofile')
    ##
    return ag.parse_args ()

if __name__ == "__main__":
    args    = get_args ()
    ####################################
    bn      = os.path.basename ( args.pkg )
    bnf     = split_extension ( bn )
    if args.ch == 'ts':
        freq_list, I, Q, U, V, I_err, Q_err, U_err, V_err = read_prepare_tscrunch (
                args.pkg,
                args.fs,
                args.nosub,
                args.v
        )
    elif args.ch == 'max':
        freq_list, I, Q, U, V, I_err, Q_err, U_err, V_err = read_prepare_max (
                args.pkg,
                args.fs,
                args.nosub,
                args.v
        )


    # print (f" 20230417: fitting to uncalibrated objects")
    # print (f" 20230417: normalize gain")
    # print (f" 20230417: by dividing by G")
    # G         = 2E3
    # I   /= G
    # Q   /= G
    # U   /= G
    # V   /= G
    # I_err /= G
    # Q_err /= G
    # U_err /= G
    # V_err /= G
    if args.frange is not None:
        flow,fhigh = args.frange
        mask  = (freq_list >= flow) & (freq_list <= fhigh)
        ## masking
        freq_list   = freq_list[mask]
        I           = I[mask]
        Q           = Q[mask]
        U           = U[mask]
        V           = V[mask]
        I_err       = I_err[mask]
        Q_err       = Q_err[mask]
        U_err       = U_err[mask]
        V_err       = V_err[mask]

    ### smooth the Stokes
    if args.bw > 0:
        Ifit  = gaussian_filter1d ( I, args.bw, axis=0 )
        Qfit  = gaussian_filter1d ( Q, args.bw, axis=0 )
        Ufit  = gaussian_filter1d ( U, args.bw, axis=0 )
        Vfit  = gaussian_filter1d ( V, args.bw, axis=0 )
    else:
        Ifit  = I
        Qfit  = Q
        Ufit  = U
        Vfit  = V

    # I_err = gaussian_filter1d ( I_err, args.bw,)
    # Q_err = gaussian_filter1d ( Q_err, args.bw,)
    # U_err = gaussian_filter1d ( U_err, args.bw,)
    # V_err = gaussian_filter1d ( V_err, args.bw,)

    ###### diagnostic plot from RMsynthesis
    if args.ofile:
        fig        = plt.figure (figsize=(11,7), dpi=300)
    else:
        fig        = plt.figure ()
    """
    data points are always errorbar


    | rm slice |
    | rm slice |

    """
    S      = np.s_[...,0] 


    # sxq,sxu    = fig.subplots ( 2, 1, sharex=True, sharey=True )
    # sxq        = fig.add_subplot (111)
    sxx,sxi,sxq,sxu,sxv = fig.subplots ( 5, 1, sharex=True, sharey=False )

    deb      = dict (marker='.', alpha=0.4, ls=':', markersize=10)
    ip       = dict (ls='-', color='k', lw=3, alpha=0.8)
    qp       = dict (ls='-', color='r', lw=3, alpha=0.8)
    up       = dict (ls='-', color='b', lw=3, alpha=0.8)
    vp       = dict (ls='-', color='g', lw=3, alpha=0.8)

    sxx.axhline (0.0, ls=':', color='k', alpha=0.4)

    # sxq        = fig.add_subplot (gs[0])
    # sxu        = fig.add_subplot (gs[1])
    # sxx.plot ( freq_list, Ifit[S], **ip  )
    if args.divi:
        sxx.plot ( freq_list, Qfit[S]/Ifit[S], **qp  )
        sxx.plot ( freq_list, Ufit[S]/Ifit[S], **up  )
        sxx.plot ( freq_list, Vfit[S]/Ifit[S], **vp  )
    else:
        sxx.plot ( freq_list, Ifit[S], **ip  )
        sxx.plot ( freq_list, Qfit[S], **qp  )
        sxx.plot ( freq_list, Ufit[S], **up  )
        sxx.plot ( freq_list, Vfit[S], **vp  )

    ### plotting
    # sxq.errorbar ( freq_list, Q[S], yerr=Q_err, color='r', **deb )
    # sxq.errorbar ( freq_list, U[S], yerr=U_err, color='b',**deb )
    sxi.errorbar ( freq_list, I[S], yerr=I_err, color='k', label='I', **deb )
    sxi.plot ( freq_list, Ifit[S], **ip  )

    sxq.errorbar ( freq_list, Q[S], yerr=Q_err, color='k', label='Q', **deb )
    sxq.plot ( freq_list, Qfit[S], **qp  )

    sxu.errorbar ( freq_list, U[S], yerr=U_err, color='k', label='U', **deb )
    sxu.plot ( freq_list, Ufit[S], **up  )

    sxv.errorbar ( freq_list, V[S], yerr=V_err, color='k', label='V', **deb )
    sxv.plot ( freq_list, Vfit[S], **vp  )

    # sxq.errorbar ( freq_list, I[S], yerr=I_err, color='k', **deb )
    # sxq.errorbar ( freq_list, Q[S]/I[S], yerr=Q_err, color='r', **deb )
    # sxq.errorbar ( freq_list, U[S]/I[S], yerr=U_err, color='b',**deb )
    # sxq.errorbar ( freq_list, V[S]/I[S], yerr=V_err, color='g',**deb )

    # sxi.legend (loc='best')
    # sxq.legend (loc='best')
    # sxu.legend (loc='best')
    # sxv.legend (loc='best')

    # for xx in [sxi, sxq, sxu, sxv]:
        # yu,yv = xx.get_ylim ()
        # xx.vlines ( freqs[zask], yu,yv, ls="-", color='k', alpha=0.4 )

    if args.divi:
        sxx.set_ylabel ('QUV/I')
    else:
        sxx.set_ylabel ('IQUV')
    sxi.set_ylabel ('I')
    sxq.set_ylabel ('Q')
    sxu.set_ylabel ('U')
    sxv.set_ylabel ('V')

    ################ beautify
    # for ix in [smu, sxq, smq, sxu]:
        # ix.axhline (0., ls='--', color='k', alpha=0.4)
        # if ix != smu:
            # smu.get_shared_x_axes().join ( ix, smu )
            # smu.get_shared_y_axes().join ( ix, smu )
            # ix.set_xticklabels ([])
        # ix.set_yticklabels ([])
        # ix.yaxis.set_label_position ('right')
        # ix.yaxis.tick_right ()

    # sxq.set_ylabel ('U')
    sxv.set_xlabel ('Freq / MHz')

    fig.suptitle (bn)
    if args.ofile:
        fig.savefig (args.ofile, dpi=300, bbox_inches='tight')
    else:
        plt.show ()
    # if rank == 0:
        # fig.savefig ( os.path.join ( args.odir, bn + ".png" ), dpi=300, bbox_inches='tight' )
        # np.savez ( os.path.join ( args.odir, bn + "_sol.npz"), **RET, **result)
    # plt.show ()
