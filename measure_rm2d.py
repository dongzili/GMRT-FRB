#!/usr/bin/python2.7
"""

modified https://gitlab.mpifr-bonn.mpg.de/nporayko/RMcalc/blob/master/RMcalc.py
modified measure_rm

2D rm fitting
Q,U over time, frequency
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
    ag   = agp.ArgumentParser ('RMcalc2d', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-m','--rmlow', default=-200, type=float, help='RM low',dest='rmlow')
    add ('-M','--rmhigh', default=-30, type=float, help='RM high', dest='rmhigh')
    add ('-s','--rmstep', default=0.1, type=float, help='RM step', dest='rmstep')
    add ('-b','--smooth', default=32, type=float, help='Gaussian smoothing sigma', dest='bw')
    add ('-f','--fscrunch', default=2, type=int, help='Frequency downsample', dest='fs')
    add ('-t','--tscrunch', default=1, type=int, help='Time downsample', dest='ts')
    add ('-j','--json', required=False, help="JSON file containing tstart,tstop,fstart,fstop", dest='json', default="calibrated_band4_bursts")
    add ('file', help="archive file")
    add ('-n','--no-subtract', help='do not subtract off', action='store_true', dest='nosub')
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-O','--outdir', help='Output directory', default='rm_measurements', dest='odir')
    ##
    return ag.parse_args ()

class QUV2D:
    """

    2D RM fitting

    Fits for RM and PA at every time bin

    NT     = number of time samples
    NF     = number of frequency samples

    CASE(1) = Simple case
    { assuming that L-fraction does not change over the duration of the burst }

    DOF    =   1     + NT   + NF
      (RM)                              +
      (PA for every time)               +
      (amplitude for every channel)

    """
    def __init__ (self, wave2, i, q, u, v, ierr, qerr, uerr, verr):
        """
        wave2: array
        stokesi: array

        data arrays are (frequency, time)
        error arrays are (frequency,)

        YFIT = {PFT} 
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
        self.paslice  = slice (1, 1+self.ts)
        self.amslice  = slice (1+self.ts, 1+self.ts+self.fs)

    def fitter (self, xfit, *arr_):
        """
        CASE(1)
        par = [rm, pa(...), amps(...)]
        """
        arr_   = np.array (arr_)
        rm     = arr_[0]
        pa     = arr_[self.paslice]
        amps   = arr_[self.amslice]
        m      = amps.reshape ((self.fs, 1)) * self.ifit * np.exp ( 2j * ( (xfit * rm) + pa.reshape ((1, self.ts)) ) )
        yy     = np.append ( np.real (m).ravel(), np.imag (m).ravel() )
        return yy

    def fit_RM (self, rms, rmlow, rmhigh):
        """
        CASE(1)
        p0  = 1 + NT + NF
        """
        p0            = np.zeros ( 1 + self.ts + self.fs )
        p0[0]         = rms
        # p0[self.amslice] = 0.1
        bounds        = (
            [rmlow]  + [-0.5*np.pi]*self.ts + [-1.0]*self.fs,
            # [rmlow]  + [-0.5*np.pi]*self.ts + [1e-3]*self.fs,
            [rmhigh] + [ 0.5*np.pi]*self.ts + [ 1.0]*self.fs
        )
        # try:
        popt, pconv   = so.curve_fit ( self.fitter, self.xfit, self.yfit, sigma=self.yerr, p0=p0, maxfev=1000000000, bounds=bounds )
        # ff            = self.fitter ( self.xfit, *popt )
        # err           = self.yfit - ff
        # adfad
        # except Exception as e:
            # print ("exception = ", e)
            # popt      = p0
            # pconv     = np.zeros ( (p0.size, p0.size) ) + 1.0
        perr          = np.sqrt ( np.diag ( pconv ) )
        self.rm, self.rmerr    = popt[0], perr[0]
        self.pa       = np.array ( popt[self.paslice] ).reshape ((1, self.ts))
        self.paerr    = np.array ( perr[self.paslice] ).reshape ((1, self.ts))
        self.amps     = np.array ( popt[self.amslice] ).reshape ((self.fs, 1))
        self.amperr   = np.array ( perr[self.amslice] ).reshape ((self.fs, 1))
        ###

    def model (self, l, i):
        m      = self.amps * i * np.exp ( 2j * ( (l.reshape((self.fs, 1)) * self.rm) + self.pa ) )
        return np.real (m), np.imag (m)

def RMsynthesis(lam2, q, u, q_err, u_err, I_sum_on, RMlow, RMhigh, RMstep, ARCH):
    """
    Based on code written by G. Heald, 9 may 2008
    """
    # subtract mean from q and u to eliminate peak at 0 rad/m^2
    q_err = np.array(q_err)
    u_err = np.array(u_err)

    # if ops.remove_baseline == "True":
    # q = q-np.average(q, weights=1./q_err**2)
    # u = u-np.average(u, weights=1./u_err**2)

    # lam2 = (299.792458/freq_list)**2
    # l02 = np.mean(lam2)
    K      = 1.0 / lam2.size
    sumI   = 1.0 / np.sum(I_sum_on)
    rmaxis = np.arange ( RMlow, RMhigh, RMstep )


    R      = np.zeros_like (rmaxis)
    FDF    = np.zeros (rmaxis.shape,dtype=np.complex128)

    for i, irm in enumerate (rmaxis):
        #Rreal = sumI*(np.sum(np.cos(-2*RM_list[i]*(lam2-l02))*I_sum_on))
        #Rimag = sumI*(np.sum(np.sin(-2*RM_list[i]*(lam2-l02))*I_sum_on))
        # FDF1 = K*(np.sum((q*np.complex(1,0)/(q_err**2)+np.complex(0,1)*u/(u_err**2))*np.exp(np.complex(0,1)*-2*irm*(lam2-l02))))
        FDF1 =  K * (
            np.sum(
                ( (q / q_err**2) + (1j * u / u_err**2) ) *  
                np.exp ( -2j * irm * (lam2-l02) )
            )
        )
        # raise RuntimeError (" the phase here is initial PA")
        #FDF1 = K*(num.sum((q*np.complex(1,0)+np.complex(0,1)*u)*np.exp(np.complex(0,1)*-2*phi[i]*(lam2-l02))))
        #R = np.append(R,np.complex(Rreal,Rimag))
        FDF[i]  = FDF1

    FDFabs = np.abs(FDF)
    imax   = np.argmax (FDFabs)
    maxFDF = rmaxis [ imax ]

    #determine error
    deltaRM = 2.*np.sqrt(3)/(np.max(lam2)-np.min(lam2))

    os.system("pam -R " + str(maxFDF) + " -FT " + ARCH + " -e RTF")
    sARCH         = split_extension ( ARCH ) + ".RTF"
    arRTF         = psrchive.Archive_load(sARCH)
    arRTF.remove_baseline()
    arRTF.centre_max_bin()
    dataRTF       = arRTF.get_data()
    Nbin          = dataRTF.shape[-1]
    I_FT_off      = dataRTF[0,0,0,int(Nbin-Nbin/10):-1]
    I_FT_off      = dataRTF[0,0,0,int(Nbin-Nbin/10):-1]
    I_FT_std      = np.std(I_FT_off)
    Q_FT          = np.array(dataRTF[0,1,0,:])
    U_FT          = np.array(dataRTF[0,2,0,:])
    L_FT_sq       = Q_FT**2+U_FT**2-I_FT_std**2
    # calculate L
    L_FT          = np.zeros_like ( Q_FT )
    for i,lft in enumerate ( L_FT_sq ):
        if lft / I_FT_std ** 2 < 2.45:
            L_FT[i] = 0
        else:
            L_FT[i] = np.sqrt ( lft )
    StoNL         = L_FT.max () / np.std ( L_FT [ofs] )
    os.system("rm " + sARCH)

    #theoretical prediction for error
    err_rmsyn     = 0.5 * deltaRM/StoNL
    # err_rmsyn = 0.5 * deltaRM
    # return RM_list, FDFabs, maxFDF, err_rmsyn, StoNL
    return rmaxis, FDFabs, maxFDF, err_rmsyn

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
    data  = ff.get_data ()
    #### making data and wts compatible
    ww = np.array (ff.get_weights ().squeeze(), dtype=bool)
    wts   = np.ones (data.shape, dtype=bool)
    wts[:,:,ww,:] = False
    mata  = np.ma.array (data, mask=wts, fill_value=np.nan)
    dd    = dict (nbin=nbin, nchan=nchan, dur=dur, fcen=fcen, fbw=fbw, data=data, wts = wts, tsamp=tsamp)
    return dd

if __name__ == "__main__":
    args    = get_args ()
    bn      = os.path.basename ( args.file )
    if not os.path.exists ( args.odir ):
        os.mkdir (args.odir)
    ##
    ## read file and ranges
    pkg     = read_ar ( args.file )

    with open (args.json, 'rb') as f:
    # with open (args.file+".json", 'rb') as f:
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
    freqs     = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen']
    freq_list = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen']

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

    freq_list = freq_list [ omask ]

    ## compute lambdas
    lam2      = np.power ( C / freq_list, 2 )
    l02       = lam2.mean ()

    if args.v:
        print (" Calling RM synthesis ... ")
    ###
    ###### RM synthesis
    ###### is run over summed on
    RMs_rmsyn, FDFabs, RM_rmsynth, err_rmsyn = RMsynthesis ( 
        lam2, 
        np.mean (Q, 1), np.mean (U, 1),
        Q_err, U_err,
        I_sum_on,
        args.rmlow, args.rmhigh, args.rmstep,
        args.file
    )
    ######
    ###
    if args.v:
        print (" done ")

    RET     = dict ()
    RET['filename'] = bn

    st  = "RM_Synthesis = {RM_rmsynth:.2f} +/- {err_rmsyn:.2f} rad/m2".format (RM_rmsynth=RM_rmsynth, err_rmsyn=err_rmsyn)

    if args.v:
        print (st)

    RET['rm_syn']     = RM_rmsynth
    RET['rmerr_syn']  = err_rmsyn

    ################################################################
    ### RM synthesis

    ### smooth the Stokes I
    Ifit  = gaussian_filter1d ( I, args.bw )

    if args.v:
        print (" Calling QU fitting ... ")

    ### do the actual call
    quv   = QUV2D ( lam2, Ifit, Q, U, V, I_err, Q_err, U_err, V_err )
    quv.fit_RM (RM_rmsynth, args.rmlow, args.rmhigh)

    if args.v:
        print (" done")

    q_model, u_model = quv.model ( lam2, I )
    
    rm_qu,rmerr_qu = quv.rm, quv.rmerr
    pa_qu          = np.rad2deg ( quv.pa )
    # pa_qu          = np.rad2deg ( np.unwrap (quv.pa - np.deg2rad (15), discont=np.pi))
    hpi            = 0.5 * np.pi
    # pa_qu          = np.rad2deg ( np.mod (quv.pa - np.deg2rad (20) + hpi, np.pi) - hpi )
    paerr_qu       = np.rad2deg ( quv.paerr )
    amp_qu         = quv.amps 
    amperr_qu      = quv.amperr

    ut  = "RM_QU = {rm:.2f} +- {rmerr:.2f} rad/m2".format ( rm = rm_qu, rmerr = rmerr_qu )

    if args.v:
        print (ut)

    jl                 = lambda x : [float(ix) for ix in x]

    RET['rm_qu']       = rm_qu
    RET['rmerr_qu']    = rmerr_qu
    RET['pa_qu']       = list (pa_qu.ravel())
    RET['paerr_qu']    = list (paerr_qu.ravel())
    RET['amp_qu']      = list (amp_qu.ravel())
    RET['amperr_qu']   = list (amperr_qu.ravel())
    ##
    RET['Ishape']      = list (I.shape)
    RET['Qshape']      = list (Q.shape)
    RET['Ushape']      = list (U.shape)
    RET['I']           = jl (I.ravel())
    RET['Q']           = jl (Q.ravel())
    RET['U']           = jl (U.ravel())
    RET['qmodel']      = jl (q_model.ravel())
    RET['umodel']      = jl (u_model.ravel())
    # RET['Ifit']        = list (Ifit)

    # dd       = block_reduce (dd_process (mata[0]), ( args.fs, 1 ), func=np.mean)
    dd       = dd_process (mata[0])

    ###### diagnostic plot from RMsynthesis
    fig        = plt.figure (figsize=(8,6), dpi=300)
    # fig        = plt.figure ()
    odict   = dict (ls='--', alpha=0.65)

    gs         = mgs.GridSpec (2, 4)

    axoq       = fig.add_subplot (gs[0,0])
    axou       = fig.add_subplot (gs[1,0])
    axiq       = fig.add_subplot (gs[0,1])
    axiu       = fig.add_subplot (gs[1,1])
    axeq       = fig.add_subplot (gs[0,2])
    axeu       = fig.add_subplot (gs[1,2])

    axp        = fig.add_subplot (gs[0,3])
    axt        = fig.add_subplot (gs[1,3])

    axp.errorbar ( times[ ons ], pa_qu[0], yerr=paerr_qu[0], color='black', ls='--', marker='s', markersize=2)

    ii      = np.abs ( amp_qu * I ).mean (0)
    ee      = np.abs ( amperr_qu * I ).mean (0)
    axt.errorbar ( times[ ons ], ii, yerr=ee, color='red', **odict )
    axt.plot ( times[ ons ], I.mean (0), color='k', **odict )
    axt.plot ( times[ ons ], V.mean (0), color='b', **odict )
    # axi.plot ( freq_list, u_model, color='k', zorder=100 )

    otimes   = times[ ons ]
    ofreqs   = freqs[ ofs ]

    RET['otimes']      = jl (otimes)

    vmin, vmax = get_vv ( Q, q_model )
    normq     = mc.Normalize ( vmin, vmax )
    axoq.imshow ( Q, aspect='auto', cmap='plasma', origin='lower', extent=[otimes.min(), otimes.max(), ofreqs[0], ofreqs[-1]], norm=normq, interpolation='none')
    axiq.imshow ( q_model, aspect='auto', cmap='plasma', origin='lower', extent=[otimes.min(), otimes.max(), ofreqs[0], ofreqs[-1]], norm=normq, interpolation='none')

    vmin, vmax = get_vv ( U, u_model )
    normu     = mc.Normalize ( vmin, vmax )
    axou.imshow ( U, aspect='auto', cmap='plasma', origin='lower', extent=[otimes.min(), otimes.max(), ofreqs[0], ofreqs[-1]], norm=normu, interpolation='none')
    axiu.imshow ( u_model, aspect='auto', cmap='plasma', origin='lower', extent=[otimes.min(), otimes.max(), ofreqs[0], ofreqs[-1]], norm=normu, interpolation='none')

    Qe        = (Q  - q_model) / Q
    Ue        = (U  - u_model) / U
    vmin, vmax = get_vv ( Qe, Ue )
    norm      = mc.Normalize ( vmin, vmax )
    # sc        = plt.cm.ScalarMappable (norm, 'coolwarm')
    axeq.imshow ( Qe, aspect='auto', cmap='coolwarm', origin='lower', extent=[otimes.min(), otimes.max(), ofreqs[0], ofreqs[-1]], norm=norm, interpolation='none')

    axeu.imshow ( Ue, aspect='auto', cmap='coolwarm', origin='lower', extent=[otimes.min(), otimes.max(), ofreqs[0], ofreqs[-1]], norm=norm, interpolation='none')

    sce = plt.cm.ScalarMappable(norm,'coolwarm')
    scq = plt.cm.ScalarMappable(normq,'plasma')
    scu = plt.cm.ScalarMappable(normu,'plasma')
    sce.set_array ([])
    scq.set_array ([])
    scu.set_array ([])

    fig.colorbar (sce, ax=[axeq, axeu], orientation='horizontal')
    fig.colorbar (scq, ax=[axoq, axiq], orientation='horizontal')
    fig.colorbar (scu, ax=[axou, axiu], orientation='horizontal')

    # for axdd in [axoq, axou, axiq, axiu]:
        # axdd.axvline ( times[ons.start], ls=':', color='k', lw=1.5 )
        # axdd.axvline ( times[ons.stop],  ls=':', color='k', lw=1.5 )
        # axdd.axhline ( freqs[ofs.start], ls=':', color='k', lw=1.5 )
        # axdd.axhline ( freqs[ofs.stop],  ls=':', color='k', lw=1.5 )


    for axo in [axoq, axiq, axou, axiu, axt]:
        axo.get_shared_x_axes ().join (axo, axp)

    for axo in [axiq, axou, axiu]:
        axo.get_shared_y_axes ().join (axo, axoq)

    axp.set_ylim ( -90, 90 )

    axoq.set_ylabel ('Q\nFreq / MHz')
    axou.set_ylabel ('U\nFreq / MHz')

    axiu.set_xlabel ('Time / ms')
    axou.set_xlabel ('Time / ms')
    axeu.set_xlabel ('Time / ms')
    axt.set_xlabel ('Time / ms')

    axp.yaxis.tick_right ()
    axp.set_ylabel ('PA / deg')
    axt.yaxis.tick_right ()
    axt.set_ylabel ('Intensity / a. u.')

    axiq.set_yticklabels ([])
    axiu.set_yticklabels ([])
    axeq.set_yticklabels ([])
    axeu.set_yticklabels ([])

    axoq.set_xticklabels ([])
    axiq.set_xticklabels ([])
    axeq.set_xticklabels ([])
    axp.set_xticklabels ([])

    axou.set_title ('Original')
    axiu.set_title ('Model')
    axeu.set_title ("Error")

    # axrm.set_xlabel ('RM [rad /m2]')
    # axrm.set_ylabel ('FDF [abs]')
    # axrm.yaxis.tick_right ()
    # axrm.yaxis.set_label_position ('right')

    # axdd.set_xlabel ('Time [ms]')
    # axdd.set_ylabel ('Frequency [MHz]')

    # axi.set_xlabel ('Frequency [MHz]')
    # axi.set_ylabel ('Stokes U')
    # axo.set_ylabel ('Stokes Q')
    # axu.set_ylabel ('Stokes L')
    # axi.yaxis.tick_right ()
    # axo.yaxis.tick_right ()
    # axu.yaxis.tick_right ()
    # axi.yaxis.set_label_position ('right')
    # axo.yaxis.set_label_position ('right')
    # axu.yaxis.set_label_position ('right')

    # axu.set_xlim ( freqs.min(), freqs.max () )

    # axdd.set_ylim ( freq_list.min (), freq_list.max () )
    tmid    = 0.5 * ( times[ons.stop]  + times[ons.start] )
    htr     = 2.5 * ( times[ons.stop]  - times[ons.start] )
    # axdd.set_xlim ( -40, 40 )
    # axdd.set_xlim  ( tmid - htr, tmid + htr )

    # axu.set_xlabel ('Frequency [MHz]')
    # axu.set_ylabel ('Stokes U')
    # axq.set_ylabel ('Stokes Q')

    fig.suptitle (bn+"\n"+st+"\n"+ut)
    fig.savefig ( os.path.join ( args.odir, bn + ".png" ), dpi=300, bbox_inches='tight' )
    with open( os.path.join ( args.odir, bn + "_sol.json" ), 'w') as f:
        json.dump ( RET, f )
    # plt.show ()
