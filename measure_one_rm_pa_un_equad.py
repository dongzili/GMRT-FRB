"""

uses ultranest

with pa
with equad


modified https://gitlab.mpifr-bonn.mpg.de/nporayko/RMcalc/blob/master/RMcalc.py
modified measure_rm

1D rm fitting
Q,U over frequency

one RM for a bunch of bursts
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

def split_extension ( f ):
    r,_ = os.path.splitext (f)
    return r

C      = 299.792458 # 1E6 * m / s 
def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('RMcalc2d_unpa', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('npz', help="npz file output by save_onerm")
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
    def __init__ (self, wave2, qi, ui, qierr, uierr):
        """
        all are (frequency,)
        """
        self.yfit     = np.append ( qi, ui )
        self.yerr     = np.append ( qierr, uierr )
        self.xfit     = wave2.copy ()
        ###
        self.rm       = None
        self.amps     = None
        self.pa       = None
        self.equad    = None
        ###
        self.qmodel   = None
        self.umodel   = None
        ###
        self.rmslice  = 0
        self.amslice  = 1
        self.paslice  = 2
        self.eqslice  = 3
        self.param_names = ["RM", "LP", "PA", "EQUAD" ] 

    def prior_transform ( self, cube ):
        """
        ultranest prior

        what should be the PA limits?
        """
        params               = np.zeros_like ( cube, dtype=np.float64 )
        params[self.rmslice] = -400.0 + (800.0 * cube[self.rmslice])
        params[self.amslice] = 3.0 * cube[self.amslice]
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
        m      = amps * np.exp ( 2j * ( (self.xfit * rm) + pa ) )
        yy     = np.append ( np.real (m).ravel(), np.imag (m).ravel() )
        return -0.5 * np.sum ( np.log ( 2.0 * np.pi * sigma2 ) ) + \
                -0.5 * np.sum ( np.power ( (self.yfit - yy) , 2 ) / sigma2 )
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
    bn      = os.path.basename ( args.npz )
    bnf     = split_extension ( bn )
    odir    = args.odir
    ####################################
    ff      = np.load ( args.npz )
    freqs = ff['freqs']
    QI  = ff['qi']
    UI  = ff['ui']
    QI_err = ff['qierr']
    UI_err = ff['uierr']
    ####################################


    ## compute lambdas
    lam2      = np.power ( C / freqs, 2 )
    l02       = np.power ( C / 650, 2 )
    lam2      -= l02
    ### wrt center of 650 MHz

    RET     = dict ()
    RET['filename'] = bn
    RET['l02']  = l02

    ### do the actual call
    quv   = QUV1D ( lam2, QI, UI, QI_err, UI_err )
    # result, rank = quv.test_fit_rm ( odir )
    result, rank = quv.fit_rm ( odir )


