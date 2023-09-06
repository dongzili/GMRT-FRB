"""

plots the np_pkl package 

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

from circ_pacv import read_pkl

def split_extension ( f ):
    r,_ = os.path.splitext (f)
    return r

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('vis_np', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('pkl', help="output by make_np")
    add ('-i','--divide-i', help='Divide by I', action='store_true', dest='divi')
    add ('-z','--zap', help='Zap the channels (comma-separated, start:stop)', dest='zap', default='')
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-o','--ofile', help='Outplot file', dest='ofile')
    ##
    return ag.parse_args ()

if __name__ == "__main__":
    args    = get_args ()
    ####################################
    base,_      = os.path.splitext ( os.path.basename ( args.pkl ) )
    pkg, freq_list, sc    = read_pkl ( args.pkl )

    freq_list  = np.ma.MaskedArray ( np.arange ( freq_list.size ), mask=freq_list.mask )
    ####################################
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
        print (f" ON-OFF Stokes-I is negative, this should not be")
        freq_list.mask[lz]      = True
        sc.mask[:,lz,:]    = True
        ff.mask[...,lz]    = True
    ## manual zapping
    for ss in args.zap.split(','):
        if len (ss) == 0:
            continue
        start, stop = ss.split(':')
        lz  = slice ( int(start), int(stop) )
        freq_list.mask[lz] = True
        sc.mask[:,lz,:]    = True
        ff.mask[...,lz]    = True

    #######################
    off_std     = sc[...,~mask].std(-1)
    #######################
    I,Q,U,V     = ff
    I_err, Q_err, U_err, V_err = off_std
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

    ### smooth the Stokes

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

    # sxq,sxu    = fig.subplots ( 2, 1, sharex=True, sharey=True )
    # sxq        = fig.add_subplot (111)
    sxx,sxi,sxq,sxu,sxv = fig.subplots ( 5, 1, sharex=True, sharey=False )

    deb      = dict (marker='.', alpha=0.4, ls=':', markersize=4)
    ip       = dict (ls='-', color='k', lw=3, alpha=0.8)
    qp       = dict (ls='-', color='r', lw=3, alpha=0.8)
    up       = dict (ls='-', color='b', lw=3, alpha=0.8)
    vp       = dict (ls='-', color='g', lw=3, alpha=0.8)

    sxx.axhline (0.0, ls=':', color='k', alpha=0.4)

    # sxq        = fig.add_subplot (gs[0])
    # sxu        = fig.add_subplot (gs[1])
    # sxx.plot ( freq_list, Ifit[S], **ip  )
    if args.divi:
        sxx.plot ( freq_list, Q/I, **qp  )
        sxx.plot ( freq_list, U/I, **up  )
        sxx.plot ( freq_list, V/I, **vp  )
    else:
        sxx.plot ( freq_list, I, **ip  )
        sxx.plot ( freq_list, Q, **qp  )
        sxx.plot ( freq_list, U, **up  )
        sxx.plot ( freq_list, V, **vp  )

    ### plotting
    # sxq.errorbar ( freq_list, Q[S], yerr=Q_err, color='r', **deb )
    # sxq.errorbar ( freq_list, U[S], yerr=U_err, color='b',**deb )
    sxi.errorbar ( freq_list, I, yerr=I_err, color='k', label='I', **deb )
    # sxi.plot ( freq_list, I, **ip  )

    sxq.errorbar ( freq_list, Q, yerr=Q_err, color='k', label='Q', **deb )
    # sxq.plot ( freq_list, Q, **qp  )

    sxu.errorbar ( freq_list, U, yerr=U_err, color='k', label='U', **deb )
    # sxu.plot ( freq_list, U, **up  )

    sxv.errorbar ( freq_list, V, yerr=V_err, color='k', label='V', **deb )
    # sxv.plot ( freq_list, V, **vp  )

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

    fig.suptitle (base)
    if args.ofile:
        fig.savefig (args.ofile, dpi=300, bbox_inches='tight')
    else:
        plt.show ()
    # if rank == 0:
        # fig.savefig ( os.path.join ( args.odir, bn + ".png" ), dpi=300, bbox_inches='tight' )
        # np.savez ( os.path.join ( args.odir, bn + "_sol.npz"), **RET, **result)
    # plt.show ()
