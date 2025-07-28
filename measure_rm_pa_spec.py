"""


"""
import os
import sys
import json

import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.colors as mc

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

def read_prepare_tscrunch ( 
        pkg_file,
        fscrunch,
        v=False
    ):
    """
    pkg_file: npz file
    fscrunch: int 
    v: bool verbose flag
    returns
    freq_list, IQUV, errors(IQUV)
    """
    ##
    pkg     = np.load ( pkg_file )

    ## read meta
    Nch     = int ( pkg['nchan'] / fscrunch )
    Nbin    = pkg['nbin']

    on_mask = np.zeros ( pkg['nbin'], dtype=bool )
    of_mask = np.ones ( pkg['nbin'], dtype=bool )
    ff_mask = np.zeros ( pkg['nchan'], dtype=bool )

    ## 20230314 : everything that is not ON is OFF
    ons     = slice ( pkg['tstart'], pkg['tstop'] )
    on_mask[ons]   = True
    of_mask[pkg['tstart']:pkg['tstop']]   = False

    ofs     = slice ( pkg['fstart'], pkg['fstop'] )
    ff_mask[ofs]   = True
    wid     = pkg['tstop'] - pkg['tstart']

    if fscrunch > 1:
        ff_mask     = np.array ( block_reduce ( ff_mask, (fscrunch,), func=np.mean ), dtype=bool )

    # read data
    data    = block_reduce (  pkg['data'][0], (1, fscrunch, 1), func=np.mean )
    wts     = np.ones (pkg['data'].shape, dtype=bool)
    ww      = np.array (pkg['wts'], dtype=bool)
    wts[:,:,ww,:] = False
    ww      = block_reduce (  wts[0] ,  (1, fscrunch, 1), func=np.mean )

    if fscrunch > 1:
        print (" Frequency downsampling by {fs:d}\t {nch0:d} --> {nch1:d}".format (fs=fscrunch, nch0=pkg['nchan'], nch1=Nch))

    # mata    = np.ma.array (data, mask=ww, fill_value=np.nan)
    mata    = data
    nsamp   = mata.shape[2]
    mask    = ww[0].sum (1) == 0.0
    zask    = ww[0].sum (1) != 0.0
    ff_mask = ff_mask & mask

    # axes
    tsamp   = float (pkg['dur']) / float ( nsamp )
    times   = np.linspace ( 0., float(pkg['dur']), nsamp )
    times   *= 1E3
    freqs     = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen']
    freq_list = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen']

    times  -= np.median (times[ons])
    btimes    = times[ons]

    freq_lo   = freq_list.min ()
    freq_hi   = freq_list.max ()

    ## Stokes ON pulse
    I_on    = np.array ( mata[0,ff_mask][...,on_mask] )
    Q_on    = np.array ( mata[1,ff_mask][...,on_mask] )
    U_on    = np.array ( mata[2,ff_mask][...,on_mask] )
    V_on    = np.array ( mata[3,ff_mask][...,on_mask] )

    ## Stokes OFF pulse
    I_off   = np.array ( mata[0,ff_mask][...,of_mask] )
    Q_off   = np.array ( mata[1,ff_mask][...,of_mask] )
    U_off   = np.array ( mata[2,ff_mask][...,of_mask] )
    V_off   = np.array ( mata[3,ff_mask][...,of_mask] )

    ## freq_list
    freq_list = freq_list [ ff_mask ]

    ## per channel std-dev
    I_std   = np.std ( I_off, 1 )
    Q_std   = np.std ( Q_off, 1 )
    U_std   = np.std ( U_off, 1 )
    V_std   = np.std ( V_off, 1 )

    ## Sum over ON pulse
    I_sum_on  = np.sum ( I_on, 1 )
    ## Choose high S/N, avoid channels with non-positive I
    omask     = np.zeros (I_sum_on.shape[0], dtype=bool)
    I_std_mask= np.std ( I_on, 1 )
    I_off_mean= np.mean (I_off, 1)
    for i,ii in enumerate (I_sum_on):
        if ( ii > 1.66 * I_std_mask[i] ) and ( ii > I_off_mean[i] ):
            omask[i] = True
    ## since i am manually selecting the subband

    I  = I_on [ omask ] -  np.mean (I_off [ omask ], 1)[:,np.newaxis]
    Q  = Q_on [ omask ] -  np.mean (Q_off [ omask ], 1)[:,np.newaxis]
    U  = U_on [ omask ] -  np.mean (U_off [ omask ], 1)[:,np.newaxis]
    V  = V_on [ omask ] -  np.mean (V_off [ omask ], 1)[:,np.newaxis]

    ## sum over time
    I      = I.sum (1)
    Q      = Q.sum (1)
    U      = U.sum (1)
    V      = V.sum (1)

    nON       = np.sqrt ( ons.stop - ons.start )
    if v:
        print (" Number of ON samples = {on:d}".format(on=ons.stop - ons.start))

    # 20230313 : use whole pulse region to compute the standard deviation
    # 20230313 : and multiply with sqrt ( width )

    I_err     = nON * I_std [ omask ]
    Q_err     = nON * Q_std [ omask ]
    U_err     = nON * U_std [ omask ]
    V_err     = nON * V_std [ omask ]
    freq_list = freq_list [ omask ]

    return freq_list, I, Q, U, V, I_err, Q_err, U_err, V_err
# from skimage.measure import block_reduce

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

    def pa_noise (self, rm_estimate, residual_power):
        """
        theoretically PA noise from Characteristic function of gaussian random distribution is 
        exp(-2sigma^2) at peak
        """
        max_mag  = np.sum ( np.exp ( 2.0j * ( self.pa - ( rm_estimate * self.w2 ) ) ) )
        max_mag  = np.abs ( max_mag ) / residual_power
        pa_sigma = np.sqrt ( -0.5 * np.log ( max_mag ) )
        return {'unbiased_paerr':pa_sigma}

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
        rpa = np.arctan ( np.tan ( self.pa - pa0 - ( rm * self.w2 ) ) )
        rpa_power = np.abs ( np.sum ( np.exp ( 2.0j * rpa ) ) )
        return rpa_power, rpa

    def chi2_reduced ( self, rm, pa0 ):
        """ chi2 reduced """
        model = self.model ( rm, pa0 )
        ye    = np.power ( self.paerr, 2 )
        # chi2  = np.sum ( np.power ( ( model - self.pa ), 2 ) / ye )
        chi2  = np.sum ( np.power ( ( model - self.pa ), 2 ))
        dof   = self.w2.size - 2
        return chi2 / dof

def split_extension ( f ):
    r,_ = os.path.splitext (f)
    return r

C      = 299.792458 # 1E6 * m / s 

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('rm_spec', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-f','--fscrunch', default=4, type=int, help='Frequency downsample', dest='fs')
    add ('-n','--ntrials', default=999, type=int, help='Number of bootstrap trials', dest='ntrials')
    add ('pkg', help="package file output by make_pkg")
    add ('-r', '--rm', help='RM range = <min-RM>:<max-RM>:<number-RM>', dest='rmg', required=True, nargs=1)
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-O','--outdir', help='Output directory', default='./', dest='odir')
    ##
    return ag.parse_args ()

if __name__ == "__main__":
    args    = get_args ()
    ####################################
    bn      = os.path.basename ( args.pkg )
    bnf     = split_extension ( bn )
    odir    = args.odir
    ####################################
    _rmg      = args.rmg[0].split(':')
    if args.v:
        print (f" RM Grid = {float(_rmg[0]):.3f} ... {float(_rmg[1]):.3f} with {int(_rmg[2]):d} steps")
    rm_grid   = np.linspace ( float(_rmg[0]), float(_rmg[1]), int(_rmg[2]) , endpoint=True )
    ####################################
    freq_list, I, Q, U, V, I_err, Q_err, U_err, V_err = read_prepare_tscrunch (
        args.pkg,
        args.fs,
        args.v
    )

    ## compute lambdas
    lam2      = np.power ( C / freq_list, 2 )

    RET     = dict ()
    CET     = dict ()
    RET['filename'] = bn
    RET['lam2'] = lam2
    RET['fs']   = args.fs
    CET['filename'] = bn
    CET['fs']   = args.fs

    if args.v:
        print (" Calling PASpec fitting ... ")

    ### do the actual call
    paspec    = PASpec ( lam2, Q, U, Q_err, U_err, I_err )

    ################################
    ### compute magnitude spectrum
    rmspec    = paspec.rm_spectrum ( rm_grid ) 

    ### fit rm 
    fitrm     = paspec.estimate_rm ( rm_grid, n_resamples=args.ntrials )

    ### fit pa
    fitpa     = paspec.estimate_pa0 ( fitrm['rm'] )

    ### get model
    model     = paspec.model ( fitrm['rm'], fitpa['pa_mean'] )
    m_model   = paspec.model ( fitrm['rm'], fitpa['pa_mean'], paspec.mw2 )
    # rpa0      = np.arctan ( np.tan ( paspec.pa - model ) )
    rpa_power, rpa     = paspec.residual_pa ( fitrm['rm'], fitpa['pa_mean'] )

    ### unbiased PA noise
    unbiased_paerr   = paspec.pa_noise ( fitrm['rm'], rpa_power )

    ### compute reduced CHI2
    rchi2     = paspec.chi2_reduced ( fitrm['rm'], fitpa['pa_mean'])

    ut    = f"RM-ML={fitrm['rm']:.3f}+-{fitrm['rm_se']:.3f}\nPA0={np.rad2deg(fitpa['pa_mean']):.3f}+-{np.rad2deg(fitpa['pa_err']):.3f}\nUnbiased-PA-error={np.rad2deg(unbiased_paerr['unbiased_paerr']):.2e}\nrCHI2={rchi2:.3f}"

    if args.v:
        print ( ut )
        print (" done")

    RET.update ( fitrm )
    RET.update ( fitpa )
    RET['w2']     = paspec.w2
    RET['pa']     = paspec.pa
    RET['paerr']  = paspec.paerr
    RET['res_pa'] = rpa
    RET['rmgrid'] = rm_grid
    CET.update ( fitrm )
    CET.update ( {k:fitpa[k] for k in ['pa_mean','pa_err']})
    CET.update ( unbiased_paerr )
    ###########################################################
    cf   = pd.DataFrame ( CET, index=[0] )

    ###########################################################
    fig = plt.figure ('paspec', figsize=(9,5))

    gs  = mgs.GridSpec ( 3, 2, figure=fig )

    axpa = fig.add_subplot ( gs[1,:] )
    axrs = fig.add_subplot ( gs[2,:], sharex=axpa )
    axgg = fig.add_subplot ( gs[0, 1] )
    axtt = fig.add_subplot ( gs[0, 0] )
    axtt.axis('off')

    axpa.errorbar ( paspec.w2, np.rad2deg( paspec.pa ), yerr=np.rad2deg( paspec.paerr ), marker='.', c='k', capsize=5, ls='' )
    axpa.plot ( paspec.mw2, np.rad2deg( m_model ), c='b' )

    axrs.plot ( paspec.w2, np.rad2deg( rpa ), marker='.', c='b' )
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

    ###########################################################
    # plt.show ()
    fig.savefig ( os.path.join ( args.odir, bn + ".png" ), dpi=300, bbox_inches='tight' )
    cf.to_csv ( os.path.join ( args.odir, bn + "_spec.csv" ), index=False )
    np.savez ( os.path.join ( args.odir, bn + "_spec.npz"), **RET)


