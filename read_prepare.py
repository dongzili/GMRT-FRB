"""
reads pkg.npz and prepares

ts_dx : does tscrunch but applies DX correction before postprocessing
DX correction is the differential gain-like term we are seeing in Cross coherence products
"""

import numpy as np

__all__ = ['read_prepare_tscrunch', 'read_prepare_max', 'read_prepare_ts_dx', 'read_prepare_peak']

def resize_slice_bool ( inp, fac):
    """
    input mask, decimation factor
    """
    ww     = block_reduce ( inp, (fac,), func=np.mean )
    return np.array ( ww, dtype=bool )

def resize_slice ( inp, fac):
    """
    input slice, input size, decimation factor
    """
    start  = int ( inp.start / fac  )
    stop   = int ( inp.stop  / fac  )
    return slice ( start, stop )

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

def read_pkg (f):
    data   = f['data'][0]
    wts    = np.ones ( data.shape, dtype=bool )
    ww     = np.array ( f['wts'], dtype=bool )
    wts[:,ww,:] = False
    return np.ma.MaskedArray ( data=data, mask=wts ),ww

def read_prepare_tscrunch ( 
        pkg_file,
        fscrunch,
        no_subtract,
        v=False
    ):
    """
    pkg_file: npz file
    fscrunch: int 
    no_subtract: bool to control whether to smooth or not
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

    if no_subtract:
        I  = I_on [ omask ]
        Q  = Q_on [ omask ]
        U  = U_on [ omask ]
        V  = V_on [ omask ]
    else:
        I  = I_on [ omask ] -  np.mean (I_off [ omask ], 1)[:,np.newaxis]
        Q  = Q_on [ omask ] -  np.mean (Q_off [ omask ], 1)[:,np.newaxis]
        U  = U_on [ omask ] -  np.mean (U_off [ omask ], 1)[:,np.newaxis]
        V  = V_on [ omask ] -  np.mean (V_off [ omask ], 1)[:,np.newaxis]

    ## sum over time
    I      = I.sum (1)[:,np.newaxis]
    Q      = Q.sum (1)[:,np.newaxis]
    U      = U.sum (1)[:,np.newaxis]
    V      = V.sum (1)[:,np.newaxis]

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

    ## 
    # print (f" 20230417: when fitting RM to uncalibrated objects")
    # print (f" 20230417: normalize gain")
    # print (f" 20230417: by dividing by G")
    # G         = 5E5
    # I   /= G
    # Q   /= G
    # U   /= G
    # V   /= G
    # I_err /= G
    # Q_err /= G
    # U_err /= G
    # V_err /= G

    return freq_list, I, Q, U, V, I_err, Q_err, U_err, V_err

def read_prepare_max ( 
        pkg_file,
        fscrunch,
        no_subtract,
        v=False
    ):
    """
    pkg_file: npz file
    fscrunch: int 
    no_subtract: bool to control whether to smooth or not
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
    ## also choose those channels its more than offpulse
    omask     = np.zeros (I_sum_on.shape[0], dtype=bool)
    I_std_mask= np.std ( I_on, 1 )
    I_off_mean= np.mean (I_off, 1)
    for i,ii in enumerate (I_sum_on):
        if ( ii > 1.66 * I_std_mask[i] ) and ( ii > I_off_mean[i] ):
            omask[i] = True
    ## since i am manually selecting the subband

    if no_subtract:
        I  = I_on [ omask ]
        Q  = Q_on [ omask ]
        U  = U_on [ omask ]
        V  = V_on [ omask ]
    else:
        I  = I_on [ omask ] -  np.mean (I_off [ omask ], 1)[:,np.newaxis]
        Q  = Q_on [ omask ] -  np.mean (Q_off [ omask ], 1)[:,np.newaxis]
        U  = U_on [ omask ] -  np.mean (U_off [ omask ], 1)[:,np.newaxis]
        V  = V_on [ omask ] -  np.mean (V_off [ omask ], 1)[:,np.newaxis]

    ## pick max
    pp     = I.mean (0)
    imax   = np.argmax ( pp )
    if v:
        print (f" profile max={pp.max():.3f} (idx = {imax:d})")
    # Q      = Q.sum (1)[:,np.newaxis]
    I      = I[...,imax][:,np.newaxis]
    Q      = Q[...,imax][:,np.newaxis]
    U      = U[...,imax][:,np.newaxis]
    V      = V[...,imax][:,np.newaxis]

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

def read_prepare_ts_dx ( 
        pkg_file,
        selfcal_file,
        fscrunch,
        no_subtract,
        v=False
    ):
    """
    pkg_file: npz file
    selfcal_file: npz file selfcal calibrator which was used to calibrate in the first place
    fscrunch: int 
    no_subtract: bool to control whether to smooth or not
    v: bool verbose flag
    returns
    freq_list, IQUV, errors(IQUV)
    """
    if selfcal_file is None:
        return read_prepare_tscrunch ( pkg_file, fscrunch, no_subtract, v )
    ##
    pkg     = np.load ( pkg_file )
    selfcal = np.load ( pkg_file )

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
    # pkg_data = pkg['data'][0]
    # data    = block_reduce (  pkg_data, (1, fscrunch, 1), func=np.mean )
    # wts     = np.ones (pkg_data.shape, dtype=bool)
    # ww      = np.array (pkg['wts'], dtype=bool)
    # wts[:,ww,:] = False
    # ww      = block_reduce (  wts ,  (1, fscrunch, 1), func=np.mean )

    mata,ww   = read_pkg ( pkg )
    sata,_    = read_pkg ( np.load (selfcal_file) )

    ##################################################
    ### DX correction
    ### mata,sata = IQUV
    #### first find ON pulse of selfcal
    pask      = sata[0].mean(0) >= 0.4 
    #### Frequency profile of ON-OFF 
    salf      = np.mean ( sata[...,pask] - sata[...,~pask], -1 )
    #### compute scaling factor
    rr        = salf[1] / salf[1].mean()
    #### reshape for easy multiplication
    rr        = rr.reshape ((-1, 1))
    ### Q -> Q / r 
    ### U -> U * r
    # mata[1]   = mata[1] / rr
    # mata[2]   = mata[2] * rr
    ### Q -> Q * r 
    ### U -> U / r
    # mata[1]   = mata[1] * rr
    # mata[2]   = mata[2] / rr
    ### Q -> Q * r 
    # mata[1]   = mata[1] * rr
    ### Q -> Q / r 
    # mata[1]   = mata[1] / rr
    ### U -> U / r 
    # mata[2]   = mata[2] / rr
    ### U -> U * r 
    mata[2]   = mata[2] * rr
    ##################################################
    ### block reduction
    mata      = block_reduce ( mata, (1, fscrunch, 1), func=np.mean )
    ##################################################

    if fscrunch > 1:
        print (" Frequency downsampling by {fs:d}\t {nch0:d} --> {nch1:d}".format (fs=fscrunch, nch0=pkg['nchan'], nch1=Nch))

    nsamp   = mata.shape[2]
    mask    = mata.mask[0].sum(1) == 0.0
    zask    = mata.mask[0].sum(1) != 0.0
    # mask    = ww[0].sum (1) == 0.0
    # zask    = ww[0].sum (1) != 0.0
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

    if no_subtract:
        I  = I_on [ omask ]
        Q  = Q_on [ omask ]
        U  = U_on [ omask ]
        V  = V_on [ omask ]
    else:
        I  = I_on [ omask ] -  np.mean (I_off [ omask ], 1)[:,np.newaxis]
        Q  = Q_on [ omask ] -  np.mean (Q_off [ omask ], 1)[:,np.newaxis]
        U  = U_on [ omask ] -  np.mean (U_off [ omask ], 1)[:,np.newaxis]
        V  = V_on [ omask ] -  np.mean (V_off [ omask ], 1)[:,np.newaxis]

    ## sum over time
    I      = I.sum (1)[:,np.newaxis]
    Q      = Q.sum (1)[:,np.newaxis]
    U      = U.sum (1)[:,np.newaxis]
    V      = V.sum (1)[:,np.newaxis]

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

    ## 
    # print (f" 20230417: when fitting RM to uncalibrated objects")
    # print (f" 20230417: normalize gain")
    # print (f" 20230417: by dividing by G")
    # G         = 5E5
    # I   /= G
    # Q   /= G
    # U   /= G
    # V   /= G
    # I_err /= G
    # Q_err /= G
    # U_err /= G
    # V_err /= G

    return freq_list, I, Q, U, V, I_err, Q_err, U_err, V_err

def read_prepare_2d ( 
        pkg_file,
        fscrunch,
        no_subtract,
        v=False
    ):
    """
    pkg_file: npz file
    fscrunch: int 
    no_subtract: bool to control whether to smooth or not
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

    if no_subtract:
        I  = I_on [ omask ]
        Q  = Q_on [ omask ]
        U  = U_on [ omask ]
        V  = V_on [ omask ]
    else:
        I  = I_on [ omask ] -  np.mean (I_off [ omask ], 1)[:,np.newaxis]
        Q  = Q_on [ omask ] -  np.mean (Q_off [ omask ], 1)[:,np.newaxis]
        U  = U_on [ omask ] -  np.mean (U_off [ omask ], 1)[:,np.newaxis]
        V  = V_on [ omask ] -  np.mean (V_off [ omask ], 1)[:,np.newaxis]


    ## 20241119 : return 2D IQUV and 1D errors

    ## sum over time
    # I      = I.sum (1)[:,np.newaxis]
    # Q      = Q.sum (1)[:,np.newaxis]
    # U      = U.sum (1)[:,np.newaxis]
    # V      = V.sum (1)[:,np.newaxis]

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

def read_prepare_tscrunch_nofcut ( 
        pkg_file,
        fscrunch,
        no_subtract,
        v=False
    ):
    """
    pkg_file: npz file
    fscrunch: int 
    no_subtract: bool to control whether to smooth or not
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

    ## 20230314 : everything that is not ON is OFF
    ons     = slice ( pkg['tstart'], pkg['tstop'] )
    on_mask[ons]   = True
    of_mask[pkg['tstart']:pkg['tstop']]   = False

    ofs     = slice ( pkg['fstart'], pkg['fstop'] )
    wid     = pkg['tstop'] - pkg['tstart']


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
    ff_mask = mask

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

    if no_subtract:
        I  = I_on [ omask ]
        Q  = Q_on [ omask ]
        U  = U_on [ omask ]
        V  = V_on [ omask ]
    else:
        I  = I_on [ omask ] -  np.mean (I_off [ omask ], 1)[:,np.newaxis]
        Q  = Q_on [ omask ] -  np.mean (Q_off [ omask ], 1)[:,np.newaxis]
        U  = U_on [ omask ] -  np.mean (U_off [ omask ], 1)[:,np.newaxis]
        V  = V_on [ omask ] -  np.mean (V_off [ omask ], 1)[:,np.newaxis]

    ## sum over time
    I      = I.sum (1)[:,np.newaxis]
    Q      = Q.sum (1)[:,np.newaxis]
    U      = U.sum (1)[:,np.newaxis]
    V      = V.sum (1)[:,np.newaxis]

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

    ## 
    # print (f" 20230417: when fitting RM to uncalibrated objects")
    # print (f" 20230417: normalize gain")
    # print (f" 20230417: by dividing by G")
    # G         = 5E5
    # I   /= G
    # Q   /= G
    # U   /= G
    # V   /= G
    # I_err /= G
    # Q_err /= G
    # U_err /= G
    # V_err /= G

    return freq_list, I, Q, U, V, I_err, Q_err, U_err, V_err

def read_prepare_peak ( 
        pkg_file,
        fscrunch,
        no_subtract,
        v=False
    ):
    """
    pkg_file: npz file made by peakpkg
    fscrunch: int 
    no_subtract: bool to control whether to smooth or not
    v: bool verbose flag
    returns
    freq_list, IQUV, errors(IQUV)
    """
    ##
    pkg     = np.load ( pkg_file )

    ## read meta
    Nch     = int ( pkg['nchan'] / fscrunch )
    Nbin    = pkg['nbin']

    # read data
    # data    = block_reduce (  pkg['data'][0], (1, fscrunch, 1), func=np.mean )
    # wts     = np.ones (pkg['data'].shape, dtype=bool)
    # ww      = np.array (pkg['wts'], dtype=bool)
    # wts[:,:,ww,:] = False
    # ww      = block_reduce (  wts[0] ,  (1, fscrunch, 1), func=np.mean )

    mata    = np.ma.MaskedArray ( pkg['max_slice'], mask=pkg['mask_max'], fill_value=np.nan )
    bata    = np.ma.MaskedArray ( pkg['min_slice'], mask=pkg['mask_min'], fill_value=np.nan )
    sata    = np.ma.MaskedArray ( pkg['std_slice'], mask=pkg['mask_std'], fill_value=np.nan )

    ## here mata, bata = (npol,nchan)

    mata    = block_reduce ( mata, (1, fscrunch), func=np.nanmean )
    bata    = block_reduce ( bata, (1, fscrunch), func=np.nanmean )
    sata    = block_reduce ( sata, (1, fscrunch), func=np.nanmean )

    if fscrunch > 1:
        print (" Frequency downsampling by {fs:d}\t {nch0:d} --> {nch1:d}".format (fs=fscrunch, nch0=pkg['nchan'], nch1=Nch))

    # mata    = np.ma.array (data, mask=ww, fill_value=np.nan)
    # mask    = ww[0].sum (1) == 0.0
    # zask    = ww[0].sum (1) != 0.0
    # ff_mask = ff_mask & mask

    # axes
    freqs     = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen']
    freq_list = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen']

    ## Stokes ON pulse
    I_on    = np.array ( mata[0] )
    Q_on    = np.array ( mata[1] )
    U_on    = np.array ( mata[2] )
    V_on    = np.array ( mata[3] )

    ## Stokes OFF pulse
    I_off   = np.array ( bata[0] )
    Q_off   = np.array ( bata[1] )
    U_off   = np.array ( bata[2] )
    V_off   = np.array ( bata[3] )

    ## freq_list
    # freq_list = freq_list [ ff_mask ]

    ## per channel std-dev
    I_std   = np.array ( sata[0] )
    Q_std   = np.array ( sata[1] )
    U_std   = np.array ( sata[2] )
    V_std   = np.array ( sata[3] )

    if no_subtract:
        I  = I_on
        Q  = Q_on
        U  = U_on
        V  = V_on
    else:
        I  = I_on - I_off
        Q  = Q_on - Q_off
        U  = U_on - U_off
        V  = V_on - V_off

    I  = I[:,np.newaxis]
    Q  = Q[:,np.newaxis]
    U  = U[:,np.newaxis]
    V  = V[:,np.newaxis]

    return freq_list, I, Q, U, V, I_std, Q_std, U_std, V_std
