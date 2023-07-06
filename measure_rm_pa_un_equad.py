"""

uses ultranest

with pa
with equad


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

from read_prepare import read_prepare_tscrunch, read_prepare_max, read_prepare_ts_dx
# from skimage.measure import block_reduce

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
    add ('-f','--fscrunch', default=2, type=int, help='Frequency downsample', dest='fs')
    add ('-c','--choice', default='ts', choices=['ts','max'], help='what kind of visualization', dest='ch')
    add ('pkg', help="package file output by make_pkg")
    # add ('-s','--selfcal', help='Selfcal file', dest='sc')
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
    def __init__ (self, wave2, i, q, u, v, ierr, qerr, uerr, verr):
        """
        wave2: array
        stokesi: array

        data arrays are (frequency, time)
        error arrays are (frequency,)

        """
        self.fs,self.ts = i.shape
        ###
        self.yfit     = np.append ( q.ravel(), u.ravel() )
        self.yerr     = np.append ( np.tile (qerr, self.ts), np.tile (uerr, self.ts) )
        self.xfit     = wave2.reshape ((self.fs, 1))
        self.ifit     = i.copy ()
        ###
        self.rm       = None
        self.pa       = None
        self.amps     = None
        ###
        self.qmodel   = None
        self.umodel   = None
        ###
        self.rmslice  = 0
        self.amslice  = 1
        self.paslice  = 2
        self.eqslice  = 3
        # self.paslice  = slice (2, 2+self.ts)
        # self.amslice  = slice (1+self.ts, 1+self.ts+self.fs)
        self.param_names = ["RM", "LP", "PA", "EQUAD" ]  #+ \
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
        params[self.amslice] = 1.0 * cube[self.amslice]
        params[self.paslice] = -0.5*np.pi + (np.pi * cube[self.paslice]) 
        params[self.eqslice] = 10.0 * cube[self.eqslice]
        return params

    def log_likelihood ( self, arr_ ):
        """
        par = [rm, amps, pa(...)]
        """
        rm     = arr_[self.rmslice]
        amps   = arr_[self.amslice]
        pa     = arr_[self.paslice]
        equad  = arr_[self.eqslice]
        sigma2 = (self.yerr**2) + (equad**2)
        # m      = amps * self.ifit * np.exp ( 2j * ( (self.xfit * rm) + pa.reshape ((1, self.ts)) ) )
        m      = amps * self.ifit * np.exp ( 2j * ( (self.xfit * rm) + pa ) )
        # m      = amps * self.ifit * np.exp ( 2j * ( (self.xfit * rm)  ) )
        yy     = np.append ( np.real (m).ravel(), np.imag (m).ravel() )
        return -0.5 * np.sum ( np.log ( 2.0 * np.pi * sigma2 ) ) + \
                -0.5 * np.sum ( np.power ( (self.yfit - yy) , 2 ) / sigma2 )
        # return -0.5 * np.sum ( np.power ( (self.yfit - yy), 2 ) )

    def test_fit_rm ( self, DIR ):
        self.rm, self.rmerr    = -114.10,0.35
        self.amps     =  0.96
        self.amperr   =  0.01
        self.pa       =  0.58
        self.paerr    =  0.01
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
            wrapped_params = [False, False, True, False],# +  [True]*self.ts,
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
        self.amps     =  popt[self.amslice] 
        self.amperr   =  perr[self.amslice] 
        self.pa       =  popt[self.paslice] 
        self.paerr    =  perr[self.paslice] 
        self.equad    =  popt[self.eqslice]
        self.equaderr =  perr[self.eqslice]
        ###
        return result, sampler.mpi_rank
        # return dict()

    def chi2_reduced ( self ):
        """ chi2 reduced """
        m     = self.amps * self.ifit * np.exp ( 2j * ( (self.xfit * self.rm) + self.pa ) )
        qm    = np.real ( m )
        um    = np.imag ( m )
        ym    = np.append ( qm, um )
        ##
        ye    = np.power ( self.yerr, 2 ) + self.equad**2
        chi2  = np.sum ( np.power ( ( ym - self.yfit ), 2 ) / ye )
        ## XXX: how do we compute CHI2?
        dof   = len (self.param_names)
        return chi2 / dof
        

    def model (self, l, i):
        # m      = self.amps * i * np.exp ( 2j * ( (l.reshape((self.fs, 1)) * self.rm) + self.pa.reshape ((1, self.ts)) ))
        m      = self.amps * i * np.exp ( 2j * ( (l.reshape((self.fs, 1)) * self.rm) + self.pa ))
        # m      = self.amps * i * np.exp ( 2j * ( (l.reshape((self.fs, 1)) * self.rm) ))
        return np.real (m), np.imag (m)

    def derotate (self, l, q, u):
        LR     = q + (1j * u)
        # m      = LR * np.exp ( (-2j * l.reshape((-1, 1)) * self.rm) )
        # m      = LR * np.exp ( -2j * ( ( l.reshape((-1, 1)) * self.rm ) + self.pa.reshape((1, self.ts)) ) )
        m      = LR * np.exp ( -2j * ( ( l.reshape((-1, 1)) * self.rm ) + self.pa ) )
        return np.real (m), np.imag (m)

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
    l02       = lam2.mean ()
    # l02       = lam2.min ()
    lam2      -= l02


    RET     = dict ()
    RET['filename'] = bn
    RET['l02']  = l02
    RET['fref'] = C / np.sqrt ( l02 )
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
        print (" Calling QU fitting ... ")

    ### do the actual call
    quv   = QUV1D ( lam2, Ifit, Q, U, V, I_err, Q_err, U_err, V_err )
    # result, rank = quv.test_fit_rm ( odir )
    result, rank = quv.fit_rm ( odir )

    if args.v:
        print (" done")

    q_rm, u_rm = quv.model ( lam2, Ifit )
    q_derot, u_derot = quv.derotate ( lam2, Q, U )
    q_res            = Qfit - q_rm
    u_res            = Ufit - u_rm
    lfraction        = quv.amps * Ifit
    chi2_red         = quv.chi2_reduced ()

    RET['freq_list'] = freq_list
    RET['model_q']   = q_rm
    RET['model_u']   = u_rm
    RET['fit_q']     = Qfit  
    RET['fit_u']     = Ufit
    RET['err_q']     = Q_err
    RET['err_u']     = U_err
    RET['lfraction'] = lfraction
    RET['derot_q']   = q_derot
    RET['derot_u']   = u_derot
    
    rm_qu,rmerr_qu = quv.rm, quv.rmerr
    # pa_qu          = np.rad2deg ( quv.pa )
    # pa_qu          = np.rad2deg ( np.unwrap (quv.pa - np.deg2rad (15), discont=np.pi))
    # hpi            = 0.5 * np.pi
    # pa_qu          = np.rad2deg ( np.mod (quv.pa - np.deg2rad (20) + hpi, np.pi) - hpi )
    pa_qu          = np.rad2deg ( quv.pa )
    paerr_qu       = np.rad2deg ( quv.paerr )
    amp_qu         = quv.amps 
    amperr_qu      = quv.amperr

    ut  = "RM_QU = {rm:.2f} +- {rmerr:.2f} rad/m2 LFRAC = {lf:.2f} +- {lferr:.2f}\nPA = {pa:.2f} +- {paerr:.2f} Chi2 reduced = {rchi2:.2f}".format ( rm = rm_qu, rmerr = rmerr_qu, lf=amp_qu, lferr=amperr_qu, pa=pa_qu, paerr=paerr_qu, rchi2=chi2_red )

    if args.v:
        print (ut)

    RET['rm_qu']       = rm_qu
    RET['rmerr_qu']    = rmerr_qu
    RET['pa_qu']       = pa_qu
    RET['paerr_qu']    = paerr_qu
    RET['amp_qu']      = amp_qu
    RET['amperr_qu']   = amperr_qu
    RET['equad']       = quv.equad
    RET['equaderr']    = quv.equaderr

    ###### diagnostic plot from RMsynthesis
    fig        = plt.figure (figsize=(11,7), dpi=300)
    # fig        = plt.figure ()
    deb      = dict (marker='.', color='k', alpha=0.5, )
    qp       = dict (ls='-', color='r', lw=3, alpha=0.7, zorder=100)
    up       = dict (ls='-', color='b', lw=3, alpha=0.7, zorder=100)
    """
    data points are always errorbar


    | rm slice |
    | rm slice |

    """
    S      = np.s_[...,0] 

    gs         = mgs.GridSpec (4, 1, wspace=0.02)

    sxq, seq, smq, sxu, seu, smu = fig.subplots ( 6, 1, sharex=True, sharey=False )

    # sxq        = fig.add_subplot (gs[0])
    # smq        = fig.add_subplot (gs[1])

    # sxu        = fig.add_subplot (gs[2])
    # smu        = fig.add_subplot (gs[3])

    ### plotting
    sxq.errorbar ( freq_list, Q[S], yerr=Q_err + quv.equad, **deb )
    sxq.plot ( freq_list, q_rm[S], **qp )

    seq.errorbar ( freq_list, q_res[S], yerr=Q_err + quv.equad, **deb)

    smq.plot ( freq_list, lfraction[S], **qp )
    smq.errorbar ( freq_list, q_derot[S], yerr=Q_err + quv.equad, **deb )

    sxu.errorbar ( freq_list, U[S], yerr=U_err + quv.equad, **deb )
    sxu.plot ( freq_list, u_rm[S], **up )

    seu.errorbar ( freq_list, u_res[S], yerr=U_err + quv.equad, **deb)

    # smu.plot ( freq_list, lfraction[S], **up )
    smu.errorbar ( freq_list, u_derot[S], yerr=U_err + quv.equad, **deb )

    ################ beautify
    for ix in [smu, sxq, smq, sxu, seq, seu]:
        ix.axhline (0., ls='--', color='k', alpha=0.4)
        # if ix != smu:
            # smu.get_shared_x_axes().join ( ix, smu )
            # smu.sharex ( ix )
            # smu.sharey ( ix )
            # smu.get_shared_y_axes().join ( ix, smu )
            # ix.set_xticklabels ([])
        # ix.set_yticklabels ([])
        ix.yaxis.set_label_position ('right')
        # ix.yaxis.tick_right ()

    smu.set_xlabel ('Freq / MHz')

    sxq.set_ylabel ('Q\nRM')
    smq.set_ylabel ('Q\nDerotated')
    sxu.set_ylabel ('U\nRM')
    seq.set_ylabel ('Q\nError')
    seu.set_ylabel ('U\nError')
    smu.set_ylabel ('U\nDerotated')

    fig.suptitle (bn+"\n"+ut)
    if rank == 0:
        fig.savefig ( os.path.join ( args.odir, bn + ".png" ), dpi=300, bbox_inches='tight' )
        np.savez ( os.path.join ( args.odir, bn + "_sol.npz"), **RET, **result)
    # plt.show ()
