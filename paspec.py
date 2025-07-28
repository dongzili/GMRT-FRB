"""
grid based approach
"""
import os
import sys
import json

import numpy as np

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.colors as mc

C      = 299.792458 # 1E6 * m / s 

class RMBootstrap:
    """
    bootstrapping RM
    """
    def __init__ (self, w2, paw2, rm_grid):
        """
        could be masked but then masked are removed
        """
        __unmasked = np.logical_not ( paw2.mask )
        self.w2    = w2 [ __unmasked ]
        self.pa    = paw2 [ __unmasked ]
        ###
        self.rm_grid = rm_grid.copy ()

    def statistic (self, w2, pas):
        """
        ML estimate of RM
        """
        ret  = [ np.abs ( np.sum ( np.exp ( 2.0j * ( pas - ( irm * w2 ) ) ) ) ) for irm in self.rm_grid ]
        return self.rm_grid [ np.argmax ( ret ) ]

    def __call__ (self, n_resamples=999, f_trial=0.85, confidence_level=0.95):
        """
        run the bootstrap

        follows
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
        but considers the wavelength2 as well.

        does alternative='two-sided' and method='basic'

        f_trial is fraction of samples per trial
        """
        ##
        ## ML estimate
        ml_estimate  = self.statistic ( self.w2, self.pa )

        nsamples     = self.w2.size
        ntrial       = int ( f_trial * nsamples )

        rng          = np.random.default_rng()

        re_stat      = np.zeros ( n_resamples, dtype=np.float32 )

        for i_resample in tqdm ( range(n_resamples), desc='Bootstrap', unit='bt' ):
            __i    = rng.choice ( nsamples, size=ntrial, replace=True, shuffle=False )
            ## slice
            t_w2   = self.w2 [ __i ]
            t_pa   = self.pa [ __i ]
            ##
            re_stat [i_resample]  = self.statistic ( t_w2, t_pa )

        alpha      = 0.5 * ( 1.0 - confidence_level )

        _ci_low    = np.percentile ( re_stat, alpha * 100. )
        _ci_high   = np.percentile ( re_stat, (1.0 - alpha) * 100 )

        ## basic
        ci_low     = (2.0*ml_estimate) - _ci_high
        ci_high    = (2.0*ml_estimate) - _ci_low

        return dict(rm=ml_estimate, rm_low=ci_low, rm_high=ci_high, rm_se=np.std(re_stat, ddof=1))

class PASpec:
    """
    PA spectrum man

    1D RM fitting

    Phi = 0.5 * arctan ( U / Q )

    Phi, Phierr = 

    RM Lambda^2 + Psi
    """
    def __init__ (self, wave2, q, u, qerr, uerr, ierr, mpoints=128):
        """
        wave2: array
        stokesi: array

        data arrays are (frequency, )
        error arrays are (frequency,)
        """
        ###
        self.l2       = np.ma.MaskedArray ( (q**2) + (u**2), mask=np.zeros_like(u, dtype=bool) )
        ### Everett, Weisburg mask
        mask          = np.sqrt( self.l2 ) / ierr < 1.57
        ###
        self.w2       = np.ma.MaskedArray ( wave2.copy(), mask=mask )
        self.w2size   = wave2.size
        self.pa       = np.ma.MaskedArray ( 0.5 * np.arctan2 ( u, q ), mask=mask )
        self.paerr    = 0.5 * np.sqrt( (q*uerr)**2 + (qerr*u)**2 ) / self.l2
        ###
        self.w2min    = wave2.min()
        self.w2max    = wave2.max()
        self.mw2      = np.linspace ( self.w2min, self.w2max, mpoints, endpoint=True )

    def rm_spectrum (self, rms):
        """
        rm spectra?
        return magnitude?
        """
        nrms = rms.size

        ret  = np.zeros ((nrms,), dtype=np.complex64)

        for irm in range ( nrms ):
            _rm  = rms[irm]
            ret[irm] = np.sum ( np.exp ( 2.0j * ( self.pa - ( _rm * self.w2 ) ) ) )

        return np.abs ( ret )

    def estimate_rm (self, rms, n_resamples=999, f_trial=0.85, confidence_level=0.95):
        """
        estimate RM error using bootstrap
        """
        boot   = RMBootstrap ( self.w2, self.pa, rms )
        res    = boot (n_resamples=n_resamples, f_trial=f_trial, confidence_level=confidence_level)
        return res

    def estimate_pa0 (self, rm):
        """
        inverse variance weighted average

        PA error is QUADRATURE sum of PA RMS and PA weighted error
        """
        pa_freq = np.arctan ( np.tan ( self.pa - (rm * self.w2) ) )
        pa_err  = self.paerr

        pa_w    = np.power ( pa_err, -2.0 )

        pa_mean = np.sum ( pa_w * pa_freq ) / np.sum ( pa_w )
        pa_mean_err  = np.power ( np.sum ( pa_w ), -0.5 )
        pa_mean_rms  = np.sqrt ( np.mean ( np.power ( pa_freq - pa_mean, 2.0 ) ) )
        pa_std       = np.sqrt ( pa_mean_rms**2 + pa_mean_err**2 )

        pa_smean= np.mean ( pa_freq )

        return dict(pa_freq=pa_freq, pa_mean=pa_mean, pa_err=pa_std, pa_mean_simple=pa_smean)

    def model (self, rm, pa0, w2=None):
        """
        pa model
        """
        if w2 is None:
            w2 = self.w2
        return np.arctan ( np.tan ( pa0 + ( rm * w2 ) ) )

    def residual_pa (self, rm, pa0):
        """
        pa model
        """
        rpa = np.arctan ( np.tan ( self.pa - pa0 - ( rm * w2 ) ) )
        rpa_power = np.abs ( np.sum ( np.exp ( 2.0j * rpa ) ) )
        return rpa_power, rpa

    def chi2_reduced ( self, rm, pa0 ):
        """ chi2 reduced """
        model = self.model ( rm, pa0 )
        ye    = np.power ( self.paerr, 2 )
        chi2  = np.sum ( np.power ( ( model - self.pa ), 2 ) / ye )
        dof   = self.w2.size - 2
        return chi2 / dof

if __name__ == "__main__":
    testpack = np.load ("sn77_testpack0.npz")
    # testpack = np.load ("sn55_testpack0.npz")

    l2 = testpack['lam2']
    l02  = testpack['l02']
    l2   += l02
    Ifit = testpack['Ifit']
    Q = testpack['Q'][...,0]
    U = testpack['U'][...,0]
    Qerr = testpack['Q_err']
    Uerr = testpack['U_err']
    Ierr = testpack['I_err']

    ###
    # lsort = np.argsort ( l2 )
    # l2    = l2[lsort]
    # Q     = Q[lsort]
    # U     = U[lsort]
    # Qerr  = Qerr[lsort]
    # Uerr  = Uerr[lsort]

    RET       = dict()

    paspec    = PASpec ( l2, Q, U, Qerr, Uerr, Ierr )
    rm_grid   = np.linspace ( -150, -50, 256, endpoint=True )

    ### compute magnitude spectrum
    rmspec    = paspec.rm_spectrum ( rm_grid ) 

    ### fit rm 
    fitrm     = paspec.estimate_rm ( rm_grid )

    ### fit pa
    fitpa     = paspec.estimate_pa0 ( fitrm['rm'] )

    ### get model
    model     = paspec.model ( fitrm['rm'], fitpa['pa_mean'] )
    m_model   = paspec.model ( fitrm['rm'], fitpa['pa_mean'], paspec.mw2 )
    # rpa0      = np.arctan ( np.tan ( paspec.pa - model ) )
    rpa_power, rpa      = paspec.residual_pa ( fitrm['rm'], fitpa['pa_mean'] )

    ### compute reduced CHI2
    rchi2     = paspec.chi2_reduced ( fitrm['rm'], fitpa['pa_mean'])

    ut    = f"RM-ML={fitrm['rm']:.3f}+-{fitrm['rm_se']:.3f}\nPA0={np.rad2deg(fitpa['pa_mean']):.3f}+-{fitpa['pa_err']:.3f}\nrCHI2={rchi2:.3f}"
    # vut   = f"peak-RM = {peak_rm:.3f}\nfit-RM = {fitopt['rm']:.3f}\nRMstd = {fitopt['rmstd']:.3f}\nPA0 = {np.rad2deg(mean_pa0):.3f}\nrCHI2={rchi2:.3f}"
    print ( ut )

    RET.update ( fitrm )
    RET.update ( fitpa )
    RET['w2']     = paspec.w2
    RET['pa']     = paspec.pa
    RET['paerr']  = paspec.paerr
    RET['res_pa'] = rpa0
    RET['rmgrid'] = rm_grid

    ###########################################################
    fig = plt.figure ('paspec')

    gs  = mgs.GridSpec ( 3, 2, figure=fig )

    axpa = fig.add_subplot ( gs[1,:] )
    axrs = fig.add_subplot ( gs[2,:], sharex=axpa )
    axgg = fig.add_subplot ( gs[0, 1] )
    axtt = fig.add_subplot ( gs[0, 0] )
    axtt.axis('off')

    axpa.errorbar ( paspec.w2, np.rad2deg( paspec.pa ), yerr=np.rad2deg( paspec.paerr ), marker='.', c='k', capsize=5, ls='' )
    axpa.plot ( paspec.mw2, np.rad2deg( m_model ), c='b' )

    axrs.plot ( paspec.w2, np.rad2deg( rpa0 ), marker='.', c='b' )
    axrs.axhline (0., ls=':', c='k', alpha=0.4 )

    # axgg.scatter ( rm_grid, rmspec, marker='.', c='k' )
    # axgg.plot ( rm_grid, rmspec_model, c='b' )
    axgg.scatter ( rm_grid, rmspec / rpa_power, marker='.', c='k' )
    # axgg.plot ( rm_grid, rmspec_model / rpa_power, c='b' )
    axgg.axvline ( fitrm['rm'], ls='--', c='b' )

    axgg.xaxis.tick_top ()
    axgg.xaxis.set_label_position('top')
    axgg.yaxis.tick_right ()
    axgg.yaxis.set_label_position('right')

    axgg.set_xlabel ('RM / rad m$^{-2}$')
    axgg.set_ylabel ('mag')
    axpa.set_ylabel ('PA / deg')
    axrs.set_ylabel ('res-PA / deg')
    axrs.set_xlabel ('Wavelength$^{2}$ / m$^{2}$')

    to_freq = lambda wav : (C / wav**0.5)
    from_freq = lambda freq: (C / freq)**2
    faxpa= axpa.secondary_xaxis ('top', functions=(to_freq, from_freq))
    faxpa.set_xlabel('Freq / MHz')

    axpa.set_xlim ( from_freq(750.), from_freq(550.) )
    axpa.set_ylim ( -90., 90. )
    axrs.set_ylim (-30, 30)

    axtt.text ( 0.5, 0.5, ut, ha='center', va='center' )

    plt.show ()

    # dpa, mpa = paline.pajumps ( -115.36, np.deg2rad(77) )

    # fig = plt.figure ('paline')


    # ax = fig.add_subplot ()

    # ax.errorbar ( paline.w2, dpa, yerr=paline.paerr, marker='.', c='k', capsize=5, ls='' )
    # ax.plot ( paline.w2, dpa, c='r' )
    # ax.plot ( paline.w2, mpa, c='b' )
    # ax.errorbar ( paline.w2, np.arctan(np.tan(dpa-mpa)), yerr=paline.paerr, marker='.', c='k', capsize=5, ls='' )


    # ax.set_xlabel('Wavelength squared')
    # ax.set_ylabel ('PA / rad')

    # plt.show ()



