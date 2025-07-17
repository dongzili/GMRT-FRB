"""

uses ultranest

with pa
with equad
like line

modified https://gitlab.mpifr-bonn.mpg.de/nporayko/RMcalc/blob/master/RMcalc.py
modified measure_rm

1D rm fitting
Q,U over frequency
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

import scipy.optimize as so

from read_prepare import read_prepare_tscrunch, read_prepare_max, read_prepare_ts_dx
# from skimage.measure import block_reduce

def split_extension ( f ):
    r,_ = os.path.splitext (f)
    return r

C      = 299.792458 # 1E6 * m / s 

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('RMcalc2d_unpa', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-b','--smooth', default=4, type=float, help='Gaussian smoothing sigma', dest='bw')
    add ('-f','--fscrunch', default=4, type=int, help='Frequency downsample', dest='fs')
    add ('-c','--choice', default='max', choices=['max', 'ts'], help='what kind of visualization', dest='ch')
    add ('pkg', help="package file output by make_pkg")
    # add ('-s','--selfcal', help='Selfcal file', dest='sc')
    add ('-n','--no-subtract', help='do not subtract off', action='store_true', dest='nosub')
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-O','--outdir', help='Output directory', default='rm_measurements', dest='odir')
    ##
    return ag.parse_args ()

class PHIL2:
    """

    1D RM fitting

    Phi = 0.5 * arctan ( U / Q )

    Phi, Phierr = 

    RM Lambda^2 + Psi
    """
    def __init__ (self, wave2, i, q, u, v, ierr, qerr, uerr, verr):
        """
        wave2: array
        stokesi: array

        data arrays are (frequency, time)
        error arrays are (frequency,)

        """
        self.fs,self.ts = i.shape
        ###
        # _phi          = np.arctan ( u / q )
        # _phierr       = np.arctan ( uerr / qerr )
        _phi          = 0.5 * np.arctan2 ( u, q )
        L2            = (q**2) + (u**2)
        _phierr       = 0.5 * np.sqrt( (q*uerr)**2 + (qerr*u)**2 ) / L2
        _phierr       = 0.5 * np.arctan2 ( uerr, qerr )
        self.yfit     = _phi.ravel()
        self.yerr     = np.tile (_phierr, self.ts)
        self.xfit     = wave2.copy()
        self.maxequad = np.sum ( self.yerr**2 )
        ###
        self.rm       = None
        self.pa       = None
        self.equad    = None
        ###
        self.phimodel = None
        ###
        self.rmslice  = 0
        self.paslice  = 1
        self.eqslice  = 2
        # self.paslice  = slice (2, 2+self.ts)
        # self.amslice  = slice (1+self.ts, 1+self.ts+self.fs)
        # self.param_names = ["RM", "PA"]  #+ \
        self.param_names = ["RM", "PA", "EQUAD" ]  #+ \
                # [f"PA_{i:02d}" for i in range (self.ts)]
                # [f"LP_{i:02d}" for i in range (self.fs)] +\

    def prior_transform ( self, cube ):
        """
        ultranest prior

        what should be the PA limits?
        """
        params               = np.zeros_like ( cube, dtype=np.float64 )
        # params[self.rmslice] = -200.0 + (400.0 * cube[self.rmslice])
        params[self.rmslice] = -400.0 + (800.0 * cube[self.rmslice])
        params[self.paslice] = -0.5*np.pi + (np.pi * cube[self.paslice]) 
        params[self.eqslice] = self.maxequad *  cube[self.eqslice]
        return params

    def log_likelihood ( self, arr_ ):
        """
        par = [rm,  pa, equad]

        phi ~ rm*lambda**2 + pa
        """
        rm     = arr_[self.rmslice]
        pa     = arr_[self.paslice]
        equad  = arr_[self.eqslice]
        sigma2 = (self.yerr**2) + (equad**2)
        # sigma2 = (self.yerr**2)
        # sigma2 = (equad**2)
        # sigma2 = (equad**2)
        ####
        # mphi   = np.arctan ( np.tan ( 2.0 * ( ( self.xfit*rm ) + pa ) ) )
        mphi   = self.fit_model ( self.xfit, rm, pa )
        ####
        return -0.5 * np.sum ( np.log ( 2.0 * np.pi * sigma2 ) ) + \
                -0.5 * np.sum ( np.power ( (self.yfit - mphi) , 2 ) / sigma2 )
        # return -0.5 * np.sum ( np.power ( (self.yfit - mphi), 2 ) )

    def test_fit_rm ( self, DIR ):
        self.rm, self.rmerr    = -112.36, 0.04
        self.pa       =  np.deg2rad( -67.96 )
        self.paerr    =  np.deg2rad (0.47)
        self.equad    = 0.70
        self.equaderr = 0.03
        ###
        return dict(), 0


    def fit_rm (self, DIR):
        """
        """
        sampler             = ultranest.ReactiveNestedSampler (
            self.param_names, 
            self.log_likelihood, self.prior_transform,
            # wrapped_params = [False, True],# +  [True]*self.ts,
            wrapped_params = [False, True, False],# +  [True]*self.ts,
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
        self.pa       =  popt[self.paslice] 
        self.paerr    =  perr[self.paslice] 
        # self.equad    =  popt[self.eqslice]
        # self.equaderr =  perr[self.eqslice]
        ###
        return result, sampler.mpi_rank

    def so_fit_rm (self, DIR):
        """
        """
        # popt, pconv = so.curve_fit ( self.fit_model, self.xfit, self.yfit, sigma=self.yerr, p0=[-115., 0.5], bounds=([-400,-0.5*np.pi],[400, 0.5*np.pi]), max_nfev=200000, verbose=2)
        popt, pconv = so.curve_fit ( self.fit_model, self.xfit, self.yfit, p0=[-115., 0.0], bounds=([-400,-0.5*np.pi],[400, 0.5*np.pi]), max_nfev=200000, verbose=2)
        perr = np.sqrt ( np.diag ( pconv ) )

        self.rm, self.rmerr    = popt[self.rmslice], perr[self.rmslice]
        self.pa       =  popt[self.paslice] 
        self.paerr    =  perr[self.paslice] 
        # self.equad    =  popt[self.eqslice]
        # self.equaderr =  perr[self.eqslice]
        ###
        return dict(), 0
        # return dict()

    def chi2_reduced ( self ):
        """ chi2 reduced """
        m     = self.model ( self.xfit )
        # ye    = np.power ( self.yerr, 2 ) + self.equad**2
        ye    = np.power ( self.yerr, 2 )
        ###
        chi2  = np.sum ( np.power ( ( m - self.yfit ), 2 ) / ye )
        ## XXX: how do we compute CHI2?
        dof   = len (self.param_names)
        return chi2 / dof

    def fit_model (self, l, rm, pa):
        # m   = np.arctan ( np.tan ( 2.0 * ( ( l*rm ) + pa ) ) )
        # m   = np.arctan ( np.tan ( 2.0 * ( ( l*rm ) + pa ) ) )
        m   = np.arctan ( np.tan ( ( l*rm ) + pa ) ) 
        return m
        
    def model (self, l):
        return self.fit_model ( l, self.rm, self.pa )

if __name__ == "__main__":
    args    = get_args ()
    ####################################
    bn      = os.path.basename ( args.pkg )
    bnf     = split_extension ( bn )
    odir    = args.odir
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

    ## compute lambdas
    lam2      = np.power ( C / freq_list, 2 )
    # l02       = lam2.mean ()
    # l02       = lam2.min ()
    # l02       = np.power ( C / 650., 2 )
    # lam2      -= l02


    RET     = dict ()
    RET['filename'] = bn
    # RET['l02']  = l02
    # RET['fref'] = C / np.sqrt ( l02 )
    RET['lam2'] = lam2

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
        print (" Calling PHI-Lambda^2 fitting ... ")

    ### do the actual call
    quv   = PHIL2 ( lam2, Ifit, Q, U, V, I_err, Q_err, U_err, V_err )
    # result, rank = quv.test_fit_rm ( odir )
    result, rank = quv.fit_rm ( odir )
    # result, rank = quv.so_fit_rm ( odir )

    if args.v:
        print (" done")

    mphi    = quv.model ( lam2 )
    mres    = np.arctan ( np.tan ( quv.yfit - mphi ) ) / quv.yerr
    chi2_red= quv.chi2_reduced ()

    RET['freq_list'] = freq_list
    RET['model_phi'] = mphi
    RET['fit_phi']   = quv.yfit
    RET['err_phi']   = quv.yerr
    
    rm_qu,rmerr_qu = quv.rm, quv.rmerr
    pa_qu          = np.rad2deg ( quv.pa )
    paerr_qu       = np.rad2deg ( quv.paerr )

    # ut  = "RM_QU = {rm:.2f} +- {rmerr:.2f} rad/m2\nPA = {pa:.2f} +- {paerr:.2f} Chi2 reduced = {rchi2:.2f}".format ( rm = rm_qu, rmerr = rmerr_qu, pa=pa_qu, paerr=paerr_qu, rchi2=chi2_red )
    ut  = "RM={rm:.2f}+-{rmerr:.2f} PA={pa:.2f}+-{paerr:.2f}".format ( rm = rm_qu, rmerr = rmerr_qu, pa=pa_qu, paerr=paerr_qu)

    if args.v:
        print (ut)

    RET['rm_l2']       = rm_qu
    RET['rmerr_l2']    = rmerr_qu
    RET['pa_l2']       = pa_qu
    RET['paerr_l2']    = paerr_qu
    # RET['equad']       = quv.equad
    # RET['equaderr']    = quv.equaderr

    ###### diagnostic plot from RMsynthesis
    fig        = plt.figure (figsize=(11,7), dpi=300)
    # fig        = plt.figure ()
    deb      = dict (marker='.', color='k', alpha=0.5,)
    qp       = dict (ls='-', color='r', lw=3, alpha=0.7, zorder=100)
    up       = dict (ls='-', color='b', lw=3, alpha=0.7, zorder=100)
    """
    data points are always errorbar

    data model residuals

    data, model

    residuals

    """

    # gs         = mgs.GridSpec (4, 1, wspace=0.02)

    xdata, xres = fig.subplots ( 2, 1, sharex=True, gridspec_kw={'hspace':0.1} )

    to_freq = lambda wav : (C / wav**0.5)
    from_freq = lambda freq: (C / freq)**2
    fdata = xdata.secondary_xaxis ('top', functions=(to_freq, from_freq))
    # fdata.set_xlim (750, 550.)
    fdata.set_xlabel ('Freq / MHz')

    # _yerr = quv.yerr + quv.equad
    _yerr = quv.yerr
    xdata.errorbar ( quv.xfit, quv.yfit, yerr=_yerr, capsize=5, ls='', **deb)
    xdata.plot ( quv.xfit, mphi, **qp )

    # xres.scatter ( lam2, mres / _yerr, **deb )
    xres.scatter ( lam2, mres, **deb )

    xres.axhline (0., ls='--', c='k', alpha=0.4)

    xres.set_xlabel ('Sq. wav / m2')

    xres.set_xlim ( from_freq(750.), from_freq(550.) )

    xres.set_ylabel ('Residual / std.dev')
    xres.set_ylim(-1., 1.)
    xdata.set_ylabel('Phi / rad')

    # fig.suptitle (bn+"\n"+ut)
    fig.suptitle (bn)
    xres.set_title (ut)
    if rank == 0:
        fig.savefig ( os.path.join ( args.odir, bn + ".png" ), dpi=300, bbox_inches='tight' )
        np.savez ( os.path.join ( args.odir, bn + "_sol.npz"), **RET, **result)
    # plt.show ()
