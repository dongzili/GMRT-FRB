"""

uses ultranest

with pa and delay

modified https://gitlab.mpifr-bonn.mpg.de/nporayko/RMcalc/blob/master/RMcalc.py
modified measure_rm

1D rm fitting
Q,U,V over frequency

delay UV rotation
dbias UV rotation phase term

20230511:
remove dbias
add equad
this is on uncalibrated archives so divide by G

remove vp
"""
import os
import sys
import json

import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.colors as mc

import ultranest
import ultranest.stepsampler

from scipy.ndimage import gaussian_filter1d

# from skimage.measure import block_reduce
from read_prepare import read_prepare_tscrunch, read_prepare_max

def split_extension ( f ):
    r,_ = os.path.splitext (f)
    return r

def block_reduce (x, fac, func=np.mean, cval=0.):
    ''' doesnt do anything with func/cval  ''' 
    xs  = x.shape
    rxs = ()
    mxs = ()
    ii  = 1
    for i, f in zip (xs, fac):
        rxs += (int(i//f), f)
        mxs += (ii,)
        ii  += 2
    # oxs = (int(xs[0]//fac[0]), int(xs[1]//fac[1]))
    dx  = x.reshape (rxs).mean (mxs)
    return dx

def get_vv ( x, y, m=1, M=3 ):
    xa     = get_v ( x, m=m, M=M )
    ya     = get_v ( y, m=m, M=M )
    vmin   = max ( xa[0], ya[0] )
    vmax   = min ( xa[1], ya[1] )
    return vmin, vmax

def get_v ( x, m=1, M=3 ):
    med  = np.median ( x )
    std  = np.std ( x )
    vmin = med - (m * std)
    vmax = med + (M * std)
    return vmin, vmax

C      = 299.792458 # 1E6 * m / s 

def dd_process (idd):
    """preprocesses filterbank"""
    nch, nbin = idd.shape
    ### remove per channel mean/std.dev
    odd    = np.float32 (idd)
    odd    -= np.mean(odd, 1).reshape ((nch, 1))
    sshape = np.std (odd,1).reshape ((nch, 1))
    odd    = np.divide (odd, sshape, out=np.zeros_like (odd), where=sshape != 0.)
    return odd

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('RMcalc2d_unpa', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-b','--smooth', default=32, type=float, help='Gaussian smoothing sigma', dest='bw')
    add ('-c','--choice', default='ts', choices=['ts','max'], help='what kind of visualization', dest='ch')
    add ('-f','--fscrunch', default=2, type=int, help='Frequency downsample', dest='fs')
    add ('pkg', help="package file output by make_pkg")
    add ('-n','--no-subtract', help='do not subtract off', action='store_true', dest='nosub')
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-O','--outdir', help='Output directory', default='rm_measurements', dest='odir')
    ##
    return ag.parse_args ()

class QUV1D:
    """

    1D RM fitting

    Fits for RM and PA at every time bin

    NT     = number of time samples
    NF     = number of frequency samples

    { Lfraction is constant over time and frequency }
    DOF    =   1     +  NT  + 1
               RM    +  PA  + LFRAC
    PARS   = { RM, LFRAC, PA(...) }

    """
    MAX_EQUAD = 1000.0
    def __init__ (self, freq, i, q, u, v, ierr, qerr, uerr, verr):
        """
        wave2: array
        stokesi: array

        data arrays are (frequency, time)
        error arrays are (frequency,)

        """
        self.fs,self.ts = i.shape
        ###
        self.yfit     = np.concatenate ( ( q.ravel(), u.ravel(), v.ravel() ) )
        self.yerr     = np.concatenate ( 
            (
            np.tile (qerr, self.ts), 
            np.tile (uerr, self.ts),
            np.tile (verr, self.ts),
            )
        )
        QUV1D.MAX_EQUAD = self.yerr.size * self.yerr.max ()
        print (f" Maximum EQUAD = {QUV1D.MAX_EQUAD:.3f}")
        self.xfreq    = freq.copy ().reshape ((self.fs, 1))
        self.xfit     = np.power ( C / self.xfreq, 2 )
        ## centering the axes
        self.fcen     = freq.mean ()
        self.xcen     = np.power ( C / self.fcen, 2 )
        self.xfreq    -= self.fcen
        self.xfit     -= self.xcen

        self.ifit     = i.copy ()
        ###
        self.rm       = None
        self.lp       = None
        self.de       = None
        self.pa       = None
        self.equad    = None
        ###
        self.qmodel   = None
        self.umodel   = None
        self.vmodel   = None
        ###
        self.rmslice  = 0
        self.lpslice  = 1
        self.deslice  = 2
        self.paslice  = 3
        self.eqslice  = 4
        self.param_names = ["RM", "LP", "DELAY_ns", "PA_rad", "EQUAD" ]  #+ \
        self.wrap_param  = [False, False, False, True, False]

    def prior_transform ( self, cube ):
        """
        ultranest prior

        what should be the PA limits?
        """
        params               = np.zeros_like ( cube, dtype=np.float64 )
        ## restrict RM range
        params[self.rmslice] = -200.0 + (400.0 * cube[self.rmslice])
        params[self.lpslice] = 1.0 * cube[self.lpslice]
        ## delay always positive
        # params[self.deslice] = 0.0 + (40.0 * cube[self.deslice]) 
        params[self.deslice] = -50.0 + (100.0 * cube[self.deslice]) 
        params[self.paslice] = -0.5*np.pi + (np.pi * cube[self.paslice]) 
        params[self.eqslice] = QUV1D.MAX_EQUAD * cube[self.eqslice]
        return params

    def log_likelihood ( self, arr_ ):
        """
        par = [rm, amps, pa(...)]

        (UV rotation) and then (QU rotation)
        this ordering matters
        """
        rm     = arr_[self.rmslice]
        lp     = arr_[self.lpslice]
        de     = arr_[self.deslice]
        pa     = arr_[self.paslice]
        eq     = arr_[self.eqslice]
        sigma2 = (self.yerr**2) + (eq**2)
        ##
        ## define angles
        ## theta=QU phi=UV
        theta  = 2.0 * ( (self.xfit * rm) + pa )
        phi    = 1.0 * ( (self.xfreq * np.pi * 1E-3 * de) )
        ## define Lp, Vp
        Lp     = lp * self.ifit
        ## there is mixing happening
        """
        see model
        """
        qq     =  Lp * np.cos ( theta )
        uu     = (Lp * np.sin ( theta ) * np.cos ( phi ))
        vv     = (Lp * np.sin ( theta ) * np.sin ( phi ))
        # qq     =  -Lp * np.sin ( theta )
        # uu     = (Lp * np.cos ( theta ) * np.cos ( phi )) 
        # vv     = (Lp * np.cos ( theta ) * np.sin ( phi ))

        yy     = np.concatenate ( ( qq.ravel(), uu.ravel(), vv.ravel() ) )
        return -0.5 * np.sum ( np.log ( 2.0 * np.pi * sigma2 ) ) + \
                -0.5 * np.sum ( np.power ( (self.yfit - yy) , 2 ) / sigma2 )

    def fit_rm (self, DIR):
        """
        """
        sampler             = ultranest.ReactiveNestedSampler (
            self.param_names, 
            self.log_likelihood, self.prior_transform,
            wrapped_params = self.wrap_param, 
            num_test_samples = 100,
            draw_multiple = True,
            num_bootstraps = 100,
            log_dir = DIR
        )
        sampler.stepsampler = ultranest.stepsampler.SliceSampler (
            nsteps = 64,
            generate_direction = ultranest.stepsampler.generate_cube_oriented_differential_direction,
            adaptive_nsteps='move-distance',
        )
        result              = sampler.run (
            min_num_live_points = 1024,
            frac_remain = 1E-4,
            min_ess = 512,
        )
        # result = dict()
        ###
        # sampler.store_tree ()
        # sampler.print_results ()
        sampler.plot_corner ()
        # sampler.plot_run ()
        # sampler.plot_trace ()
        ###
        popt          = result['posterior']['median']
        perr          = result['posterior']['stdev']
        # popt          = np.array ([-107.45, 0.95] + list(np.random.random(self.ts)))
        # perr          = np.array ([0.01, 0.001]   + list(1E-2 *np.random.random(self.ts)))

        self.rm, self.rmerr    = popt[self.rmslice], perr[self.rmslice]
        self.lp, self.lperr    =  popt[self.lpslice], perr[self.lpslice]
        self.pa       =  popt[self.paslice] 
        self.paerr    =  perr[self.paslice] 
        self.de       =  popt[self.deslice]
        self.deerr    =  perr[self.deslice]
        self.eq       =  popt[self.eqslice] 
        self.eqerr    =  perr[self.eqslice] 
        ###
        return result, sampler.mpi_rank
        # return dict()

    def test_fit_rm (self, DIR):
        """
        """
        # self.rm, self.rmerr    = -0.7, 1.0
        self.rm, self.rmerr    = -116.59,1.39
        self.lp     =  0.88
        self.lperr  =  0.01
        self.pa       =  np.deg2rad ( -54.15 )
        self.paerr    =  np.pi/10
        self.de       =  31.466
        self.deerr    =  0.143
        self.eq       =  2E-3
        self.eqerr    =  1E-4
        ###
        return dict(), 0

    def model (self, freq, i):
        ## centering
        fcen   = freq.mean ()
        lcen   = np.power ( C / fcen, 2 )
        ## reshaping
        f      = freq.copy().reshape ((self.fs, 1))
        l2     = np.power ( C / f, 2 )
        ## centering
        f      -= fcen
        l2     -= lcen
        ##
        ## define angles
        ## theta=QU phi=UV
        theta  = 2.0 * ( (l2 * self.rm) + self.pa )
        phi    = 1.0 * ( (f * np.pi * 1E-3 * self.de) )
        ## define Lp, Vp
        Lp     = self.lp * self.ifit
        ## there is mixing happening
        """
        the sign of RM is wrong

        so make the IQUV=(10Lp0) correction
        like we did for lin_pacv_c1
        """
        qq     =  Lp * np.cos ( theta )
        uu     = (Lp * np.sin ( theta ) * np.cos ( phi )) 
        vv     = (Lp * np.sin ( theta ) * np.sin ( phi ))
        # qq     =  -Lp * np.sin ( theta )
        # uu     = (Lp * np.cos ( theta ) * np.cos ( phi )) 
        # vv     = (Lp * np.cos ( theta ) * np.sin ( phi ))
        ##
        yy     = np.concatenate ( ( qq, uu, vv ) )
        return qq, uu, vv

    def derotate_QU (self, f, q, u):
        QU     = q + (1.0j * u)
        l      = np.power ( C / f, 2 )
        m      = QU * np.exp ( -2j * ( ( l.reshape((-1, 1)) * self.rm ) + self.pa ) )
        return np.real (m), np.imag (m)

    def derotate_UV (self, f, u, v):
        UV     = u + (1.0j * v)
        m      = UV * np.exp ( -1j * ( ( f.reshape((-1, 1)) * self.de * np.pi * 1E-3 ) ))
        return np.real (m), np.imag (m)

if __name__ == "__main__":
    args    = get_args ()
    ####################################
    bn      = os.path.basename ( args.pkg )
    bnf     = split_extension ( bn )
    odir    = args.odir

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

    ##
    ## deciding with negating Q
    ## because that is physical
    ## Q is like XX-YY
    ## maybe the X,Y convention is swapped
    # print (f" 20231010: swapping Q sign")
    # Q = -Q
    # print (f" 20231010: swapping U sign")
    # U = -U

    # print (f" 20230417: fitting to uncalibrated objects")
    # print (f" 20230417: normalize gain")
    # print (f" 20230417: by dividing by G")
    # G         = 2E5
    # I   /= G
    # Q   /= G
    # U   /= G
    # V   /= G
    # I_err /= G
    # Q_err /= G
    # U_err /= G
    # V_err /= G

    ## compute lambdas
    lam2      = np.power ( C / freq_list, 2 )
    l02       = lam2.mean ()
    # l02       = lam2.min ()
    lam2      -= l02

    RET     = dict ()
    RET['filename'] = bn
    RET['l02']  = l02
    RET['fref'] = C / np.sqrt ( l02 )
    RET['lam2'] = lam2

    RET['freq_list'] = freq_list

    RET['fs']   = args.fs
    RET['nosub'] = args.nosub
    RET['ch']   = args.ch

    ### smooth the Stokes
    if args.bw > 0:
        Ifit  = gaussian_filter1d ( I, args.bw, axis=1 )
        Qfit  = gaussian_filter1d ( Q, args.bw, axis=1 )
        Ufit  = gaussian_filter1d ( U, args.bw, axis=1 )
        Vfit  = gaussian_filter1d ( V, args.bw, axis=1 )
    else:
        Ifit  = I
        Qfit  = Q
        Ufit  = U
        Vfit  = V

    if args.v:
        print (" Calling QU fitting ... ")

    ### do the actual call
    quv   = QUV1D ( freq_list, Ifit, Q, U, V, I_err, Q_err, U_err, V_err )
    # quv   = QUV1D ( freq_list, Ifit, Q, V, U, I_err, Q_err, V_err, U_err )
    result, rank = quv.fit_rm ( odir )
    # result, rank = quv.test_fit_rm ( odir )

    if args.v:
        print (" done")

    q_rm, u_rm, v_rm = quv.model ( freq_list, Ifit )
    # q_derotqu, u_derotqu = quv.derotate_QU ( freq_list, Q, U )
    # u_derotuv, v_derotuv = quv.derotate_UV ( freq_list, u_derotqu, V )
    lfraction        = quv.lp * I
    
    rm_qu,rmerr_qu = quv.rm, quv.rmerr
    pa_qu          = np.rad2deg ( quv.pa )
    paerr_qu       = np.rad2deg ( quv.paerr )
    eq_qu          = quv.eq
    eqerr_qu       = quv.eqerr
    lp_qu          = quv.lp
    lperr_qu       = quv.lperr
    de_qu          = quv.de 
    deerr_qu       = quv.deerr

    ut  = "RM_QU = {rm:.2f} +- {rmerr:.2f} rad/m2  PA = {pa:.2f} +- {paerr:.2f}\nLFRAC = {lf:.2f} +- {lferr:.2f}\nDelay = {de:.3f} +- {deerr:.3f} ns  EQUAD = {db:.3f} +- {dberr:.3f}".format ( rm = rm_qu, rmerr = rmerr_qu, lf=lp_qu, lferr=lperr_qu, de=de_qu, deerr=deerr_qu, db=eq_qu, dberr=eqerr_qu, pa=pa_qu, paerr=paerr_qu)

    if args.v:
        print (ut)

    RET['rm_qu']       = rm_qu
    RET['rmerr_qu']    = rmerr_qu
    RET['pa_qu']       = pa_qu
    RET['paerr_qu']    = paerr_qu
    RET['equad_qu']       = eq_qu
    RET['equaderr_qu']    = eqerr_qu
    RET['lp_qu']      = lp_qu
    RET['lperr_qu']   = lperr_qu
    RET['de_qu']       = de_qu
    RET['deerr_qu']    = deerr_qu

    ###### diagnostic plot from RMsynthesis
    fig        = plt.figure (figsize=(11,7), dpi=300)
    # fig        = plt.figure ()
    # idict    = dict (aspect='auto', cmap='coolwarm', origin='lower', interpolation='none', extent=[btimes[0], btimes[-1], freqs[0], freqs[-1]])
    deb      = dict (marker='.', color='k', alpha=0.4)
    qp       = dict (ls='-', color='r', lw=3, alpha=0.8)
    up       = dict (ls='-', color='b', lw=3, alpha=0.8)
    vp       = dict (ls='-', color='g', lw=3, alpha=0.8)
    """
    data points are always errorbar


    | rm slice |
    | rm slice |

    """
    S      = np.s_[...,0] 

    # gs         = mgs.GridSpec (6, 1, wspace=0.02)
    # sxq        = fig.add_subplot (gs[0])
    # srq        = fig.add_subplot (gs[1])
    # sdq        = fig.add_subplot (gs[2])
    # sxu        = fig.add_subplot (gs[3])
    # sru        = fig.add_subplot (gs[4])
    # sdu        = fig.add_subplot (gs[5])

    sxq,sxu,sxv  = fig.subplots (3, 1, sharex=True, sharey=True)

    ### plotting
    sxq.errorbar ( freq_list, Q[S], yerr=Q_err, **deb )
    sxq.plot ( freq_list, q_rm[S], **qp )

    sxu.errorbar ( freq_list, U[S], yerr=U_err, **deb )
    sxu.plot ( freq_list, u_rm[S], **up )

    sxv.errorbar ( freq_list, V[S], yerr=V_err, **deb )
    sxv.plot ( freq_list, v_rm[S], **vp )


    sxv.set_xlabel ('Freq / MHz')

    sxq.set_ylabel ('Q')
    sxu.set_ylabel ('U')
    sxv.set_ylabel ('V')

    fig.suptitle (bn+"\n"+ut)
    if rank == 0:
        fig.savefig ( os.path.join ( args.odir, bn + ".png" ), dpi=300, bbox_inches='tight' )
        np.savez ( os.path.join ( args.odir, bn + "_sol.npz"), **RET, **result)
        # plt.show ()
