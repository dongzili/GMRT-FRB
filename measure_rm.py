#!/usr/bin/python2.7
"""

modified https://gitlab.mpifr-bonn.mpg.de/nporayko/RMcalc/blob/master/RMcalc.py
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
    ag   = agp.ArgumentParser ('RMcalc', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-m','--rmlow', default=-200, type=float, help='RM low',dest='rmlow')
    add ('-M','--rmhigh', default=-30, type=float, help='RM high', dest='rmhigh')
    add ('-s','--rmstep', default=0.1, type=float, help='RM step', dest='rmstep')
    add ('-b','--smooth', default=32, type=float, help='Gaussian smoothing sigma', dest='bw')
    add ('-f','--fscrunch', default=2, type=int, help='Frequency downsample', dest='fs')
    add ('-j','--json', required=False, help="JSON file containing tstart,tstop,fstart,fstop", dest='json', default="calibrated_band4_bursts")
    add ('file', help="archive file")
    add ('-n','--no-subtract', help='do not subtract off', action='store_true', dest='nosub')
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-O','--outdir', help='Output directory', default='rm_measurements', dest='odir')
    ##
    return ag.parse_args ()

class QUVD:
    """

    RM fitting

    Fits for RM, PA, Delay and maybe tau

    """
    def __init__ (self, freq, i, q, u, v, ierr, qerr, uerr, verr):
        """
        freq: MHz
        wave2: array
        stokesi: array
        """
        self.n        = freq.size
        self.ss       = slice ( 0, self.n )
        ###
        self.yfit     = np.append ( q, u )
        self.yerr     = np.append ( qerr, uerr )
        # self.xfit     = np.append ( wave2, wave2 )
        # self.ifit     = np.append ( i, i )
        self.nfit     = freq 
        self.ifit     = i.copy ()
        ###
        self.rm       = None
        self.pa       = None
        self.delay    = None
        self.amps     = None
        ###
        self.qmodel   = None
        self.umodel   = None

    def fitter (self, nfit, rm, pa, delay, *amps_):
        """
        par = [rm, pa, delay, amps(...)]
        """
        lfit     = np.power ( C / nfit, 2 )
        # amps   = np.append ( amps_, amps_ )
        amps   = np.array (amps_)
        m      = amps * self.ifit * np.exp ( 2j * ( (lfit * rm) + (nfit * 1E6 * np.pi * delay) + pa ) )
        # m      = amps * self.ifit * np.exp ( 2j * ( (lfit * rm) + pa ) )
        yy     = np.append ( np.real (m), np.imag (m) )
        return yy

    def fit_RM (self, rms, rmlow, rmhigh):
        p0            = np.zeros ( 3 + self.n )
        p0[0]         = rms
        p0[2]         = 1e-8
        bounds        = (
            [rmlow, -0.5*np.pi, -np.inf]  + [-1.0]*self.n,
            [rmhigh, 0.5*np.pi, np.inf] + [1.0]*self.n
        )
        # try:
        popt, pconv   = so.curve_fit ( self.fitter, self.nfit, self.yfit, sigma=self.yerr, p0=p0, maxfev=1000000, bounds=bounds )
        # ff            = self.fitter ( self.xfit, *popt )
        # err           = self.yfit - ff
        # adfad
        # except Exception as e:
            # print ("exception = ", e)
            # popt      = p0
            # pconv     = np.zeros ((self.n+2,self.n+2))
        perr          = np.sqrt ( np.diag ( pconv ) )
        self.rm, self.rmerr    = popt[0], perr[0]
        self.pa, self.paerr    = popt[1], perr[1]
        self.delay, self.delayerr = popt[2], perr[2]
        self.amps     = np.array ( popt[3:] )
        self.amperr   = np.array ( perr[3:] )
        ###

    def model (self, f, i):
        l             = np.power ( C / f, 2 )
        m             = self.amps * i * np.exp ( 2j * ( (l * self.rm) +  (f * 1E6 * self.delay * np.pi) + self.pa ) )
        return np.real (m), np.imag (m)

class QUV:
    """

    RM fitting

    Fits for RM, PA and maybe tau

    """
    def __init__ (self, wave2, i, q, u, v, ierr, qerr, uerr, verr):
        """
        wave2: array
        stokesi: array
        """
        self.n        = wave2.size
        self.ss       = slice ( 0, self.n )
        ###
        self.yfit     = np.append ( q, u )
        self.yerr     = np.append ( qerr, uerr )
        # self.xfit     = np.append ( wave2, wave2 )
        # self.ifit     = np.append ( i, i )
        self.xfit     = wave2.copy ()
        self.ifit     = i.copy ()
        ###
        self.rm       = None
        self.pa       = None
        self.amps     = None
        ###
        self.qmodel   = None
        self.umodel   = None

    def fitter (self, xfit, rm, pa, *amps_):
        """
        par = [rm, pa, amps(...)]
        """
        # amps   = np.append ( amps_, amps_ )
        amps   = np.array (amps_)
        m      = amps * self.ifit * np.exp ( 2j * ( (xfit * rm) + pa ) )
        yy     = np.append ( np.real (m), np.imag (m) )
        return yy

    def fit_RM (self, rms, rmlow, rmhigh):
        p0            = np.zeros ( 2 + self.n )
        p0[0]         = rms
        bounds        = (
            [rmlow, -0.5*np.pi] + [-1.0]*self.n,
            [rmhigh, 0.5*np.pi] + [1.0]*self.n
        )
        # try:
        popt, pconv   = so.curve_fit ( self.fitter, self.xfit, self.yfit, sigma=self.yerr, p0=p0, maxfev=1000000, bounds=bounds )
        # ff            = self.fitter ( self.xfit, *popt )
        # err           = self.yfit - ff
        # adfad
        # except Exception as e:
            # print ("exception = ", e)
            # popt      = p0
            # pconv     = np.zeros ((self.n+2,self.n+2))
        perr          = np.sqrt ( np.diag ( pconv ) )
        self.rm, self.rmerr    = popt[0], perr[0]
        self.pa, self.paerr    = popt[1], perr[1]
        self.amps     = np.array ( popt[2:] )
        self.amperr   = np.array ( perr[2:] )
        ###

    def model (self, l, i):
        m             = self.amps * i * np.exp ( 2j * ( (l * self.rm) + self.pa ) )
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
    dur   = ff.get_first_Integration().get_duration()
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
    dd    = dict (nbin=nbin, nchan=nchan, dur=dur, fcen=fcen, fbw=fbw, data=data, wts = wts)
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
    off_s   = slice (0, 100)


    if args.fs > 1:
        ofs = resize_slice ( ofs, args.fs )

    ## read meta
    Nbin    = pkg['nbin']
    Nch     = int ( pkg['nchan'] / args.fs )

    # read data
    data    = block_reduce (  pkg['data'][0], (1, args.fs, 1), func=np.mean )
    ww      = block_reduce (  pkg['wts'][0], (1, args.fs, 1), func=np.mean )

    if args.fs > 1 and args.v:
        print (" Downsampling by {fs:d}\t {nch0:d} --> {nch1:d}".format (fs=args.fs, nch0=pkg['nchan'], nch1=Nch))

    mata    = np.ma.array (data, mask=ww, fill_value=np.nan)
    nsamp   = mata.shape[2]
    mask    = ww[0].sum (1) == 0.0

    # axes
    tsamp   = float (pkg['dur']) / float (pkg['nbin'])
    times   = np.arange (nsamp) * tsamp
    times   -= times[nsamp//2]
    times   *= 1E3
    freqs     = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen']
    freq_list = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen']

    freq_lo   = freq_list.min ()
    freq_hi   = freq_list.max ()

    ## Stokes ON pulse
    I_on    = np.transpose ( mata[0,ofs,ons] )
    Q_on    = np.transpose ( mata[1,ofs,ons] )
    U_on    = np.transpose ( mata[2,ofs,ons] )
    V_on    = np.transpose ( mata[3,ofs,ons] )

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
    I_sum_on  = np.sum ( I_on, 0 )
    Q_sum_on  = np.sum ( Q_on, 0 )
    U_sum_on  = np.sum ( U_on, 0 )
    V_sum_on  = np.sum ( V_on, 0 )



    ## Choose high S/N, avoid channels with non-positive I
    omask     = np.zeros (I_sum_on.shape, dtype=bool)
    for i,ii in enumerate (I_sum_on):
        if ii > 2 * I_std[i] :
            omask[i] = True
    ## since i am manually selecting the subband

    if not args.nosub:
        I  = I_sum_on [ omask ]
        Q  = Q_sum_on [ omask ]
        U  = U_sum_on [ omask ]
        V  = V_sum_on [ omask ]
    else:
        I  = I_sum_on [ omask ] - np.sum ( I_off [ omask ], 1 )
        Q  = Q_sum_on [ omask ] - np.sum ( Q_off [ omask ], 1 )
        U  = U_sum_on [ omask ] - np.sum ( U_off [ omask ], 1 )
        V  = V_sum_on [ omask ] - np.sum ( V_off [ omask ], 1 )

    L  = np.sqrt ( Q**2 + U**2 )

    nON       = np.sqrt ( ons.stop - ons.start )
    if args.v:
        print ("Number of ON samples = {on:d}".format(on=ons.stop - ons.start))

    I_err     = nON * I_std [ omask ]
    Q_err     = nON * Q_std [ omask ]
    U_err     = nON * U_std [ omask ]
    V_err     = nON * V_std [ omask ]

    freq_list = freq_list [ omask ]

    ## compute lambdas
    lam2      = np.power ( C / freq_list, 2 )
    l02       = lam2.mean ()

    if args.v:
        print (" Calling RM synthesis ... ")
    ###
    ###### RM synthesis
    RMs_rmsyn, FDFabs, RM_rmsynth, err_rmsyn = RMsynthesis ( 
        lam2, 
        Q, U, Q_err, U_err, 
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

    DELAY = False

    Ifit  = gaussian_filter1d ( I, args.bw )

    if DELAY:
        quv   = QUVD ( freq_list, Ifit, Q, U, V, I_err, Q_err, U_err, V_err )
    else:
        quv   = QUV ( lam2, Ifit, Q, U, V, I_err, Q_err, U_err, V_err )
    quv.fit_RM (RM_rmsynth, args.rmlow, args.rmhigh)

    if DELAY:
        q_model, u_model = quv.model ( freq_list, Ifit )
    else:
        q_model, u_model = quv.model ( lam2, Ifit )
    
    rm_qu,rmerr_qu = quv.rm, quv.rmerr
    pa_qu,paerr_qu = quv.pa, quv.paerr
    amp_qu         = quv.amps 
    amperr_qu      = quv.amperr
    if DELAY:
        delay_qu, delayerr_qu = quv.delay, quv.delayerr

    if DELAY:
        ut  = "RM_QU = {rm:.2f} +- {rmerr:.2f} rad/m2\nDelay = {delay_qu:.3f} ns".format ( rm = rm_qu, rmerr = rmerr_qu, delay_qu = delay_qu * 1E9 )
    else:
        ut  = "RM_QU = {rm:.2f} +- {rmerr:.2f} rad/m2".format ( rm = rm_qu, rmerr = rmerr_qu )

    if args.v:
        print (ut)

    RET['rm_qu']       = rm_qu
    RET['rmerr_qu']    = rmerr_qu
    RET['pa_qu']       = pa_qu
    RET['paerr_qu']    = paerr_qu
    RET['amp_qu']      = list (amp_qu)
    RET['amperr_qu']   = list (amperr_qu)
    # RET['Ifit']        = list (Ifit)
    if DELAY:
        RET['delay_qu']    = delay_qu
        RET['delayerr_qu'] = delayerr_qu

    dd       = block_reduce (dd_process (mata[0]), ( args.fs, 1 ), func=np.mean)

    ###### diagnostic plot from RMsynthesis
    fig        = plt.figure (figsize=(8,6), dpi=300)
    # fig        = plt.figure ()

    gs         = mgs.GridSpec (4, 2)

    axdd       = fig.add_subplot (gs[3,0])
    axrm       = fig.add_subplot (gs[3,1])

    axo        = fig.add_subplot (gs[0,:])
    axi        = fig.add_subplot (gs[1,:])
    axu        = fig.add_subplot (gs[2,:])

    axrm.step ( RMs_rmsyn, FDFabs, where='mid', color='k', lw=1 )
    axrm.axvline ( RM_rmsynth, ls=':', color='k', lw=2 )
    # axrm.text ( 0.99, 0.99, st, ha='right', va='top', transform=axrm.transAxes )

    odict   = dict (ls='--', alpha=0.65)
    axo.errorbar ( freq_list, Q, yerr=Q_err, color='red',**odict )
    axo.plot ( freq_list, q_model, color='k', zorder=100 )
    # axo.text ( 0.05, 0.99, ut, ha='left', va='top', transform=axo.transAxes )

    axi.errorbar ( freq_list, U, yerr=U_err, color='blue', **odict )
    axi.plot ( freq_list, u_model, color='k', zorder=100 )

    # axi.errorbar ( freq_list, amp, yerr=V_err, color='gren', **odict )

    ii      = np.abs ( amp_qu * I )
    axu.errorbar ( freq_list, ii, yerr=amperr_qu * I, color='green', **odict )
    # axi.plot ( freq_list, u_model, color='k', zorder=100 )

    med  = np.median (dd)
    std  = np.std (dd)
    vmin = med - 1 * std
    vmax = med + 3 * std
    axdd.imshow ( dd, aspect='auto', cmap='plasma', origin='lower', extent=[times.min(), times.max(), freqs[0], freqs[-1]], vmin=vmin, vmax=vmax)


    axdd.axvline ( times[ons.start], ls=':', color='k', lw=1.5 )
    axdd.axvline ( times[ons.stop],  ls=':', color='k', lw=1.5 )
    axdd.axhline ( freqs[ofs.start], ls=':', color='k', lw=1.5 )
    axdd.axhline ( freqs[ofs.stop],  ls=':', color='k', lw=1.5 )

    axo.get_shared_x_axes ().join (axo, axi)
    axo.get_shared_x_axes ().join (axo, axu)
    axo.get_shared_y_axes ().join (axo, axi)

    axrm.set_xlabel ('RM [rad /m2]')
    axrm.set_ylabel ('FDF [abs]')
    axrm.yaxis.tick_right ()
    axrm.yaxis.set_label_position ('right')

    axdd.set_xlabel ('Time [ms]')
    axdd.set_ylabel ('Frequency [MHz]')

    axi.set_xlabel ('Frequency [MHz]')
    axi.set_ylabel ('Stokes U')
    axo.set_ylabel ('Stokes Q')
    axu.set_ylabel ('Stokes L')
    axi.yaxis.tick_right ()
    axo.yaxis.tick_right ()
    axu.yaxis.tick_right ()
    axi.yaxis.set_label_position ('right')
    axo.yaxis.set_label_position ('right')
    axu.yaxis.set_label_position ('right')

    axu.set_xlim ( freqs.min(), freqs.max () )

    # axdd.set_ylim ( freq_list.min (), freq_list.max () )
    tmid    = 0.5 * ( times[ons.stop]  + times[ons.start] )
    htr     = 2.5 * ( times[ons.stop]  - times[ons.start] )
    # axdd.set_xlim ( -40, 40 )
    axdd.set_xlim  ( tmid - htr, tmid + htr )

    # axu.set_xlabel ('Frequency [MHz]')
    # axu.set_ylabel ('Stokes U')
    # axq.set_ylabel ('Stokes Q')

    fig.suptitle (bn+"\n"+st+"\n"+ut)
    fig.savefig ( os.path.join ( args.odir, bn + ".png" ), dpi=300, bbox_inches='tight' )
    with open( os.path.join ( args.odir, bn + "_sol.json" ), 'w') as f:
        json.dump ( RET, f )
    # plt.show ()

