"""

Takes uncalibrated RM corrected pulsar scan.
Make a pkg out of it.

Fit for cable delay, pa, linear-pol-amp

20220922: this works! the delay measurements are sensible

dudeeee, need to compensate for Ionospheric RM and parallactic angle
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaahhhhhhhhhhhhhhh
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

from read_prepare import read_prepare_tscrunch, read_prepare_max, read_prepare_ts_dx

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('measure_crossdelay', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-b','--smooth', default=32, type=float, help='Gaussian smoothing sigma', dest='bw')
    add ('-f','--fscrunch', default=2, type=int, help='Frequency downsample', dest='fs')
    add ('-c','--choice', default='ts', choices=['ts','max'], help='what kind of visualization', dest='ch')
    add ('pkg', help="package file output by make_pkg")
    add ('-n','--no-subtract', help='do not subtract off', action='store_true', dest='nosub')
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-O','--outdir', help='Output directory', default='rm_measurements', dest='odir')
    return ag.parse_args ()

def split_extension ( f ):
    r,_ = os.path.splitext (f)
    return r

class MeasureDelay:
    """

    1D Delay, PA, AMP fitting

    Three parameters only

    """
    def __init__ (self, freq, i, q, u, v, ierr, qerr, uerr, verr):
        """
        freq: array
        stokesi: array

        data arrays are  (frequency,)
        error arrays are (frequency,)

        YFIT = {Q,U} 
        """
        self.fs       = i.shape[0]
        ###
        self.yfit     = np.append ( q, u )
        self.yerr     = np.append ( qerr, uerr )
        self.xfit     = np.array (freq)
        self.ifit     = i.copy ()
        ###
        self.yerr2    = np.power ( self.yerr, 2 )
        self.nterm    = -0.5 * np.sum ( np.log ( 2.0 * np.pi * self.yerr2 ) )
        ###
        self.delay_pi = None
        self.amps     = None
        self.pa       = None
        #
        self.param_names = ["DELAY_PI", "LP", "PA" ] 
        self.deslice  = 0
        self.amslice  = 1
        self.paslice  = 2

    def prior_transform ( self, cube ):
        """
        ultranest prior

        what should be the PA limits?
        """
        params               = np.zeros_like ( cube, dtype=np.float64 )
        params[self.deslice] = -400.0 + (800.0 * cube[self.deslice])
        params[self.amslice] = 1.0 * cube[self.amslice]
        params[self.paslice] = -0.5*np.pi + (np.pi * cube[self.paslice]) 
        return params

    def log_likelihood ( self, arr_ ):
        """
        par = [delay_pi, amps, pa]
        """
        dpi    = arr_[self.deslice]
        amps   = arr_[self.amslice]
        pa     = arr_[self.paslice]
        ####
        m      = amps * self.ifit * np.exp         \
            (                                      \
                (1j * self.xfit * 1E-3  * dpi) +   \
                (1j * pa)                          \
            )
        ####
        yy     = np.append ( np.real (m), np.imag (m) )
        return -0.5 * np.sum ( np.power ( (self.yfit - yy) , 2 ) / self.yerr2 ) +\
                self.nterm

    def fit_delay (self, DIR):
        """
        """
        sampler             = ultranest.ReactiveNestedSampler (
            self.param_names, 
            self.log_likelihood, self.prior_transform,
            wrapped_params = [False, False, True],
            num_test_samples = 100,
            draw_multiple = True,
            num_bootstraps = 100,
            log_dir = DIR
        )
        sampler.stepsampler = ultranest.stepsampler.SliceSampler (
            nsteps = 25,
            generate_direction = ultranest.stepsampler.generate_cube_oriented_differential_direction,
            adaptive_nsteps='move-distance',
        )
        result              = sampler.run (
            min_num_live_points = 1024,
            frac_remain = 1E-4,
            min_ess = 512,
        )
        sampler.plot_corner ()
        ###
        popt          = result['posterior']['median']
        perr          = result['posterior']['stdev']
        self.delay_pi, self.delay_pierr = popt[self.deslice], perr[self.deslice]
        self.amps     =  popt[self.amslice] 
        self.amperr   =  perr[self.amslice] 
        self.pa       =  popt[self.paslice] 
        self.paerr    =  perr[self.paslice] 
        ###
        return result

    def test_fit_delay ( self, DIR ):
        self.delay_pi = 35.72
        self.delay_pierr = 0.01
        self.amps     =  0.84
        self.amperr   =  0.01
        self.pa       =  -0.03
        self.paerr    =  0.01
        ###
        return dict()

    def model (self, f, i):
        """ forward  """
        m      = self.amps * i * np.exp              \
            (                                        \
                (1j * f * 1E-3  * self.delay_pi) +   \
                (1j * self.pa)                       \
            )
        return np.real (m), np.imag (m)

    def de_model (self, f, q, u):
        """ inverse -- removing the affect of delay """
        lr     = (q) + (1.0j*u)
        m      = lr * np.exp                           \
            (                                          \
                (-1j * f * 1E-3  * self.delay_pi)   +  \
                (-1j * self.pa)                        \
            )
        return np.real (m), np.imag (m)

if __name__ == "__main__":
    args    = get_args ()
    ####################################
    bn      = os.path.basename ( args.pkg )
    bnf     = split_extension ( bn )
    odir    = args.odir
    if not os.path.exists ( args.odir ):
        os.mkdir (args.odir)
    ## prepare files
    solfile     = os.path.join ( args.odir, bnf + ".delay_sol.npz" )
    pngfile     = os.path.join ( args.odir, bnf + ".delay_sol.png" )
    ####################################
    if args.ch == 'ts':
        freq_list, I, Q, U, V, I_err, Q_err, U_err, V_err = read_prepare_tscrunch (
        # freq_list, I, Q, U, V, I_err, Q_err, U_err, V_err = read_prepare_ts_dx (
                args.pkg,
                # args.sc,
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
    S        = np.s_[...,0] 
    ####################################
    RET     = dict ()
    RET['filename'] = bn
    RET['fs']       = args.fs
    RET['nosub']    = args.nosub
    RET['ch']       = args.ch

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
        print (" Measuring delay ... ")

    quv    = MeasureDelay ( freq_list, Ifit[S], Q[S], U[S], V[S], I_err, Q_err, U_err, V_err )
    res    = quv.fit_delay ( odir )
    # res    = quv.test_fit_delay  ( odir )
    RET.update ( res )
    RET['freq']  = freq_list
    RET['dpi']   = quv.delay_pi
    RET['bias']  = quv.pa
    RET['dpierr']   = quv.delay_pierr
    RET['biaserr']  = quv.paerr

    if args.v:
        print (" done")

    q_model, u_model = quv.model ( freq_list, Ifit[S] )
    q_de, u_de       = quv.de_model ( freq_list, Q[S], U[S] )
    q_res            = Q[S] - q_model
    u_res            = U[S] - u_model
    
    delay, delay_err = quv.delay_pi / np.pi, quv.delay_pierr / np.pi
    pa,pae           = np.rad2deg ( [quv.pa, quv.paerr] )
    amps, amperr     = quv.amps, quv.amperr

    ut  = "Delay = {d:.3f} +- {de:.3f} ns\nPA = {pa:.2f} +- {pae:.2f} deg\nLfraction = {a:.3f} +- {ae:.3f}".format ( d = delay, de = delay_err, pa=pa, pae=pae, a=amps, ae=amperr)

    if args.v:
        print (ut)

    ## store in RET
    RET['delay_ns']      = delay
    RET['delayerr_ns']   = delay_err
    RET['pa_deg']        = quv.pa
    RET['paerr_deg']     = quv.paerr
    RET['amp']           = amps
    RET['amperr']        = amperr

    ###### diagnostic plot
    """
    fitted parameters are just numbers
    they show up in title text

    for Q,U against frequency
    | data,forward |
    | data-forward |
    | inverse-data |
    """
    fig      = plt.figure (figsize=(8,6), dpi=300)
    odict    = dict (ls='--', alpha=0.65)
    deb      = dict (marker='.', color='k', alpha=0.5, )
    qp       = dict (ls='-', color='r', lw=3, alpha=0.7, zorder=100)
    up       = dict (ls='-', color='b', lw=3, alpha=0.7, zorder=100)

    # qxf, qxe, qxi, uxf, uxe, uxi = fig.subplots ( 6, 1, sharex=True, sharey=True )

    # ( qxf, qxe, qxi ), ( uxf, uxe, uxi ) = fig.subplots ( 3, 2, sharex=True, sharey='col')
    ( (qxf,uxf), (qxe,uxe), (qxi,uxi) ) = fig.subplots ( 3, 2, sharex=True, sharey='row')

    qxf.errorbar ( freq_list, Q[S], yerr=Q_err, **deb )
    qxf.plot ( freq_list, q_model, **qp )

    qxe.errorbar ( freq_list, q_res, **deb )

    qxi.plot ( freq_list, q_de, **qp )

    uxf.errorbar ( freq_list, U[S], yerr=U_err, **deb )
    uxf.plot ( freq_list, u_model, **up )

    uxe.errorbar ( freq_list, u_res, **deb )

    uxi.plot ( freq_list, u_de, **up )

    uxi.set_xlabel ('Freq / MHz')
    qxi.set_xlabel ('Freq / MHz')

    # qxf.set_ylabel ('Q')
    # qxe.set_ylabel ('Q\nError')
    # qxi.set_ylabel ('Q\nUn-Delay')

    qxf.set_ylabel ('Q,U')
    qxe.set_ylabel ('Error\nQ,U')
    qxi.set_ylabel ('Un-Delay\nQ,U')

    # uxf.set_ylabel ('U')
    # uxe.set_ylabel ('U\nError')
    # uxi.set_ylabel ('U\nUn-Delay')

    # fig.suptitle (bn+"\n"+ut)
    fig.suptitle (ut)

    fig.savefig ( pngfile, dpi=300, bbox_inches='tight' )
    np.savez ( solfile, **RET )
