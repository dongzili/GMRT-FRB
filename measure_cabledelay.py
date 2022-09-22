#!/usr/bin/python2.7
"""

take uncal psr scan.
rm correct with true value.

now fit for delay on a single time slice

now add PA

fit for delay, PA, amps

only one amp per frequency

20220922: this works! the delay measurements are sensible

modified from measure_rm2d

"""
from __future__ import print_function

import os
import sys
import json

import numpy as np

import psrchive

import matplotlib
matplotlib.use ('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.colors as mc

import scipy.optimize as so

from scipy.ndimage import gaussian_filter1d

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
    ag   = agp.ArgumentParser ('measure_delay', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-b','--smooth', default=8, type=float, help='Gaussian smoothing sigma', dest='bw')
    add ('-f','--fscrunch', default=2, type=int, help='Frequency downsample', dest='fs')
    add ('-t','--tscrunch', default=1, type=int, help='Time downsample', dest='ts')
    add ('-j','--json', required=False, help="JSON file containing tstart,tstop,fstart,fstop", dest='json', default="calibrated_band4_bursts")
    add ('-s','--step', help="time slice to choose in between 0 and 1", dest='slice', default=None)
    add ('file', help="archive file")
    add ('-n','--no-subtract', help='do not subtract off', action='store_true', dest='nosub')
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-O','--outdir', help='Output directory', default='rm_measurements', dest='odir')
    ##
    return ag.parse_args ()

class MeasureDelay:
    """

    1D Delay, RM, Pa fitting

    Fits for Delay

    NT     = number of time samples
    NF     = number of frequency samples

    CASE(1) = using solution from measure_rm2d
    { assuming that L-fraction does not change over the duration of the burst }

    DOF    =   1     +    1
      (Delay)            +
      (RM)            

    CASE(2) = keep RM constant
    and choose the known RM
    DOF    = 1

    CASE(3) = uncal, RM corrected file is input
    time slice to use is also input

    DOF  = 1 (Delay) + amps (NF,)

    """
    def __init__ (self, freq, i, q, u, v, ierr, qerr, uerr, verr):
        """
        freq: array
        stokesi: array

        data arrays are (frequency, )
        error arrays are (frequency,)

        YFIT = {PFT} 
        """
        self.fs       = i.shape[0]
        ###
        self.yfit     = np.append ( q, u )
        self.yerr     = np.append ( qerr, uerr )
        self.xfit     = np.array (freq)
        self.ifit     = i.copy ()
        self.weight   = np.power (self.yerr, -1)
        self.weight   /= self.weight.sum()
        ###
        self.delay    = None
        self.amps     = None
        ###

    # def fitter (self, xfit, delay, pa, *amps_):
    def fitter (self, arr_):
        """
        CASE(1)
        par = [delay, pa, amps]
        """
        delay  = arr_[0]
        pa     = arr_[1]
        amps   = arr_[2]
        m      = amps * self.ifit * \
            np.exp (                                       \
                (                                          \
                    (2j * self.xfit * 1E-3 * delay * np.pi)   +      \
                    (2j * pa)                                     \
                )                                          \
            )
        yy     = np.append ( np.real (m), np.imag (m) )
        # ee     = ( yy - self.yfit ) / self.yerr
        # ee     =  yy - self.yfit
        # ee     = np.abs ((self.yfit - yy) / self.yerr)
        # ee     = np.abs ((self.yfit - yy))
        # ee     = (self.yfit - yy) * self.weight
        ee     = (self.yfit - yy)
        # print (" error = {m:.2e} +- {s:.2e}".format (m = np.mean (ee), s = np.std (ee)))
        return ee

    def fit_delay (self, idelay):
        """
        CASE(1)
        p0  = 1 + 1

        CASE(2)
        p0  = 1
        """
        p0            = np.zeros ( 1 + 1 + 1 )
        p0[0]         = idelay
        p0[1]         = np.pi / 4.
        p0[2]         = 1.0
        # p0[0]         = 
        # bounds        = (
            # [ -np.inf, -np.inf, -0.5 * np.pi]  + [-1.0]*self.fs, 
            # [  np.inf,  np.inf,  0.5 * np.pi]  + [1.0]*self.fs
            # [  -30, -0.5 * np.pi, -np.inf],
            # [   30,  0.5 * np.pi, np.inf]
        # )
        bounds        = ( -np.inf, np.inf )
        # popt, pconv   = so.curve_fit ( self.fitter, self.xfit, self.yfit, sigma=self.yerr, p0=p0, maxfev=1000000000, bounds=bounds )
        res           = so.least_squares ( self.fitter, p0, bounds=bounds, max_nfev=1000000, verbose=2, jac='3-point', 
                tr_solver='exact',
                loss='linear',
                method='trf',
                ftol=1E-6
        )
        popt          = res.x
        print (res.message)
        pconv         = np.linalg.inv ( res.jac.T.dot (res.jac) )
        perr          = np.sqrt ( np.diag ( pconv ) )
        # self.delay, self.delayerr    = np.power (10, [popt[0], perr[0]])
        self.delay, self.delayerr    = popt[0], perr[0]
        self.pa, self.paerr          = popt[1], perr[1]
        self.amps, self.ampserr      = popt[2], perr[2]
        ###

    def model (self, f, i):
        m      = self.amps * i * np.exp                               \
            (                                                     \
                (2j * f * 1E-3  * self.delay * np.pi) +   \
                (2j * self.pa)                            \
            )
        return np.real (m), np.imag (m)

def read_ar (fname, remove_baseline=True):
    """
    reads
    """
    ff    = psrchive.Archive_load (fname)
    ff.convert_state ('Stokes')
    if remove_baseline:
        ff.remove_baseline ()
    ff.dedisperse ()
    ###
    nbin  = ff.get_nbin()
    nchan = ff.get_nchan()
    # dur   = ff.get_first_Integration().get_duration()
    dur   = ff.get_first_Integration().get_folding_period ()
    fcen  = ff.get_centre_frequency ()
    fbw   = ff.get_bandwidth ()
    fchan = fbw / nchan
    tsamp = dur / nbin
    ###
    start_time   = ff.start_time ().in_days ()
    end_time     = ff.end_time ().in_days ()
    mid_time     = 0.5 * ( start_time + end_time )
    ###
    src          = ff.get_source ()
    rmc          = ff.get_faraday_corrected ()
    pcal         = ff.get_poln_calibrated ()
    if not rmc:
        raise RuntimeError (" Must perform RM correction before calling")
    if pcal:
        raise RuntimeError (" Must be uncalibrated")
    ###
    data  = ff.get_data ()
    #### making data and wts compatible
    ww = np.array (ff.get_weights ().squeeze(), dtype=bool)
    wts   = np.ones (data.shape, dtype=bool)
    wts[:,:,ww,:] = False
    mata  = np.ma.array (data, mask=wts, fill_value=np.nan)
    dd    = dict (nbin=nbin, nchan=nchan, dur=dur, fcen=fcen, fbw=fbw, data=data, wts = wts, tsamp=tsamp, src=src, obstime=mid_time)
    return dd

if __name__ == "__main__":
    args    = get_args ()
    bn      = os.path.basename ( args.file )
    if not os.path.exists ( args.odir ):
        os.mkdir (args.odir)
    ##
    RET     = dict ()
    ## read file and ranges
    pkg     = read_ar ( args.file )
    RET['filename']     = bn
    RET['source']       = pkg['src']
    RET['obstime']      = pkg['obstime']

    # with open (args.json, 'rb') as f:
    with open (args.file+".json", 'r') as f:
        ran = json.load (f)

    def resize_slice ( inp, fac):
        """
        input slice, input size, decimation factor
        """
        start  = int ( inp.start / fac  )
        stop   = int ( inp.stop  / fac  )
        return slice ( start, stop )

    ons     = slice ( ran['tstart'], ran['tstop'] )
    ofs     = slice ( ran['fstart'], ran['fstop'] )
    off_s   = slice (0, ran['tstop'] - ran['tstart'])


    if args.fs > 1:
        ofs   = resize_slice ( ofs, args.fs )
    if args.ts > 1:
        ons   = resize_slice ( ons, args.ts )
        off_s = resize_slice ( off_s, args.ts )


    ## read meta
    Nch     = int ( pkg['nchan'] / args.fs )
    Nbin    = int ( pkg['nbin'] / args.ts )

    # read data
    data    = block_reduce (  pkg['data'][0], (1, args.fs, args.ts), func=np.mean )
    ww      = block_reduce (  pkg['wts'][0],  (1, args.fs, args.ts), func=np.mean )

    if args.fs > 1 and args.v:
        print (" Frequency downsampling by {fs:d}\t {nch0:d} --> {nch1:d}".format (fs=args.fs, nch0=pkg['nchan'], nch1=Nch))
    if args.ts > 1 and args.v:
        print (" Time downsampling by {fs:d}\t {nch0:d} --> {nch1:d}".format (fs=args.ts, nch0=pkg['nbin'], nch1=Nbin))

    mata    = np.ma.array (data, mask=ww, fill_value=np.nan)
    nsamp   = mata.shape[2]
    mask    = ww[0].sum (1) == 0.0

    # axes
    tsamp   = float (pkg['dur']) / float ( nsamp )
    times   = np.linspace ( 0., float(pkg['dur']), nsamp )
    times   *= 1E3
    foff    = pkg['fbw'] / Nch
    half_off  = 0.5 * foff
    # freqs     = np.linspace (  )
    freqs     = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen'] #- half_off
    freq_list = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen'] #- half_off
    # print (" mid frequency = {f:.3f} MHz".format (f=freq_list[Nch//2]))
    # adfa

    times  -= np.median (times[ons])

    freq_lo   = freq_list.min ()
    freq_hi   = freq_list.max ()

    ## Stokes ON pulse
    I_on    = mata[0,ofs,ons]
    Q_on    = mata[1,ofs,ons]
    U_on    = mata[2,ofs,ons]
    V_on    = mata[3,ofs,ons]

    ## Stokes OFF pulse
    I_off   = np.array ( mata[0,ofs,off_s] )
    Q_off   = np.array ( mata[1,ofs,off_s] )
    U_off   = np.array ( mata[2,ofs,off_s] )
    V_off   = np.array ( mata[3,ofs,off_s] )

    ## freq_list
    freq_list = freq_list [ ofs ]

    ## per channel std-dev
    I_std   = np.std ( I_off, 1 )
    Q_std   = np.std ( Q_off, 1 )
    U_std   = np.std ( U_off, 1 )
    V_std   = np.std ( V_off, 1 )


    ## Sum over ON pulse
    I_sum_on  = np.sum ( I_on, 1 )
    Q_sum_on  = np.sum ( Q_on, 1 )
    U_sum_on  = np.sum ( U_on, 1 )
    V_sum_on  = np.sum ( V_on, 1 )

    ## Choose high S/N, avoid channels with non-positive I
    omask     = np.zeros (I_sum_on.shape[0], dtype=bool)
    I_std     = np.std ( I_on, 1 )
    for i,ii in enumerate (I_sum_on):
        if ii > 1.66 * I_std[i] :
            omask[i] = True
    ## since i am manually selecting the subband

    if not args.nosub:
        I  = I_on [ omask ]
        Q  = Q_on [ omask ]
        U  = U_on [ omask ]
        V  = V_on [ omask ]
    else:
        I  = I_on [ omask ] -  I_off [ omask ]
        Q  = Q_on [ omask ] -  Q_off [ omask ]
        U  = U_on [ omask ] -  U_off [ omask ]
        V  = V_on [ omask ] -  V_off [ omask ]

    nON       = np.sqrt ( ons.stop - ons.start )
    if args.v:
        print (" Number of ON samples = {on:d}".format(on=ons.stop - ons.start))

    # XXX : no longer need to multiply

    I_err     = I_std [ omask ]
    Q_err     = Q_std [ omask ]
    U_err     = U_std [ omask ]
    V_err     = V_std [ omask ]

    if args.slice:
        OOO     = int ((args.slice * (ran['tstop'] - ran['tstart'])))
    else:
        OOO     = I.mean (0).argmax ()
    # ons     = OOO
    if args.v:
        print (" ON Time samples = {a:d}, slice at {c:d}".format( a = ran['tstop'] - ran['tstart'], c = OOO ))

    I         = I [ ..., OOO ]
    Q         = Q [ ..., OOO ]
    U         = U [ ..., OOO ]
    V         = V [ ..., OOO ]

    freq_list = freq_list [ omask ]

    ## compute lambdas
    lam2      = np.power ( C / freq_list, 2 )
    l02       = lam2.mean ()

    ### smooth the Stokes I
    Ifit  = gaussian_filter1d ( I, args.bw )
    # Ifit  = I.copy()

    # Q         = Q / Ifit
    # U         = U / Ifit
    # V         = V / Ifit
    # Ifit      = I / Ifit
    # print (Ifit, Q, U, V)


    if args.v:
        print (" Measuring delay ... ")

    # def save_json (f, **kwargs):
        # jl                 = lambda x : [float(ix) for ix in x]
        # RET  = {k:jl (v) for k,v in kwargs.items()}
        # with open (f, "w") as ff:
            # json.dump (RET, ff)

    ### do the actual call
    # save_json ("lmao_psr_why.json", freq=freq_list, I=Ifit, Q=Q, U=U, V=V)
    # adfadf
    quv    = MeasureDelay ( freq_list, Ifit, Q, U, V, I_err, Q_err, U_err, V_err )
    quv.fit_delay ( 10 )

    if args.v:
        print (" done")

    q_model, u_model = quv.model ( freq_list, Ifit )
    
    delay, delay_err = quv.delay, quv.delayerr
    pa,pae           = np.rad2deg ( [quv.pa, quv.paerr] )
    amps, amperr     = quv.amps, quv.ampserr

    ut  = "Delay = {d:.3f} +- {de:.3f} ns\nPA = {pa:.2f} +- {pae:.2f} deg\nLfraction = {a:.3f} +- {ae:.3f}".format ( d = delay, de = delay_err, pa=pa, pae=pae, a=amps, ae=amperr)

    if args.v:
        print (ut)

    # sys.exit (0)

    jl                 = lambda x : [float(ix) for ix in x]

    RET['delay_ns']      = delay
    RET['delayerr_ns']   = delay_err
    RET['pa_deg']        = pa
    RET['paerr_deg']     = pae
    RET['amp']           = amps
    RET['amperr']        = amperr

    # dd       = block_reduce (dd_process (mata[0]), ( args.fs, 1 ), func=np.mean)
    dd       = dd_process (mata[0])
    pp       = dd.mean (0)

    ###### diagnostic plot from RMsynthesis
    fig        = plt.figure (figsize=(8,6), dpi=300)
    # fig        = plt.figure ()
    odict   = dict (ls='--', alpha=0.65)

    gs         = mgs.GridSpec (6, 2)

    axpp       = fig.add_subplot (gs[0,0])
    axtt       = fig.add_subplot (gs[0,1])

    axee       = fig.add_subplot (gs[1,:])
    axqq       = fig.add_subplot (gs[2,:])
    axuu       = fig.add_subplot (gs[3,:])
    axvv       = fig.add_subplot (gs[4,:])
    axii       = fig.add_subplot (gs[5,:])

    # axll       = axvv.twinx ()

    axpp.plot ( times[ons], pp[ons], color='black', ls='-', marker='s', markersize=2)
    axpp.axvline ( times[ ons ][OOO], ls='--', color='red' )

    axtt.axis ('off')

    e2   = quv.fitter ( [quv.delay, quv.pa, quv.amps] )
    qe   = e2[:quv.fs]
    ue   = e2[quv.fs:]
    axee.plot ( freq_list, qe, color='red', markersize=2, marker='s', label='Q' , alpha=0.5, lw=1)
    axee.plot ( freq_list, ue, color='blue', markersize=2, marker='s', label='U', alpha=0.5, lw=1)
    axee.legend (loc='best')

    axqq.errorbar ( freq_list, Q, yerr=Q_err, ls='', marker='s', color='k', alpha=0.4, markersize=2 )
    axqq.plot ( freq_list, q_model, ls='-', color='red', alpha=0.6 )
    axqq.plot ( freq_list, Ifit, ls=':', color='green', alpha=0.5 )

    axuu.errorbar ( freq_list, U, yerr=U_err, ls='', marker='s', color='k', alpha=0.4, markersize=2 )
    axuu.plot ( freq_list, u_model, ls='-', color='blue', alpha=0.6)
    axuu.plot ( freq_list, Ifit, ls=':', color='green', alpha=0.5 )

    axii.plot ( freq_list, Ifit, ls=':', color='green',label='Ifit', lw=3)
    axii.errorbar ( freq_list, I, yerr=I_err, ls='', marker='s', color='k', alpha=0.9, markersize=0.8, label='I', elinewidth=0.2 )
    ll    = np.sqrt ( Q**2 + U**2 )
    axii.errorbar ( freq_list, ll, ls='', marker='s', color='r', alpha=0.9, markersize=1, label='L', )
    axii.legend (loc='best')

    axvv.errorbar ( freq_list, np.zeros_like (freq_list) + amps, yerr=np.zeros_like (freq_list) + amperr, ls='-', marker='s', color='k', alpha=0.9, markersize=2 )
    ll   = ll / Ifit
    axvv.errorbar ( freq_list, ll, ls='', marker='s', color='b', alpha=0.9, markersize=2 )
    # axvv.errorbar ( freq_list, V, yerr=V_err, ls='', marker='s', color='k', alpha=0.4, markersize=2 )
    # axvv.plot ( freq_list, v_model, ls='-', color='green', alpha=0.6)

    for iax in [axuu, axuu, axii, axee]:
        axqq.get_shared_x_axes().join (axqq, iax)
        axqq.get_shared_y_axes().join (axqq, iax)

    RTS  = "Uncalibrated, RM corrected (RM = literature value)\ncolor=Stokes (model)\nblack points = data"
    axtt.text (0.5, 0.5, "\n".join ([RTS, ut]), ha='center', va='center', )

    axee.set_ylabel ('Error')
    axqq.set_ylabel ('Q')
    axuu.set_ylabel ('U')
    axii.set_ylabel ('I')
    axvv.set_ylabel ('Lfraction')

    for iax in [axqq, axuu, axvv, axee, axii]:
        iax.yaxis.set_label_position('right')

    axii.set_xlabel ('Freq / MHz')
    axtt.set_xlabel ('Time / ms')

    axtt.set_ylabel ('Intensity / a. u.')

    # axqq.set_yticklabels ([])
    # axee.set_yticklabels ([])
    # axuu.set_yticklabels ([])
    # axii.set_yticklabels ([])
    # axvv.set_yticklabels ([])
    axtt.set_yticklabels ([])

    axee.set_xticklabels ([])
    axqq.set_xticklabels ([])
    axvv.set_xticklabels ([])
    axuu.set_xticklabels ([])

    fig.suptitle (bn)
    fig.savefig ( os.path.join ( args.odir, bn + "_delay.png" ), dpi=300, bbox_inches='tight' )
    with open( os.path.join ( args.odir, bn + "_delay7.json" ), 'w') as f:
        json.dump ( RET, f )
    # plt.show ()
