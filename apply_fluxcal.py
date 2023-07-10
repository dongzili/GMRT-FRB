"""
apply_fluxcal

take fluxcal solution output from :make_fluxcal.py:
apply it to burst pkg.npz ( calibrated burst )
which is used for RM fitting :make_pkg.py:


outputs a stokes-I flux calibrated time series (frequency averaged)
with ON phase in time/frequency

actual energetics would be measured from :measure_ffw.py:
    ffw is flux, fluence, width
    also measures frequency extent

actually, let me just measure here to save some intermediate steps
"""

import os
import json

import warnings

import numpy as np

import datetime as dt
from   dateutil import tz

import matplotlib.pyplot as plt

import astropy.time  as at
import astropy.units as au
import astropy.coordinates as asc

#############################################
def get_args ():
    import argparse
    agp = argparse.ArgumentParser ("apply_fluxcal", description="Applies fluxcal solution", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add ('pkg', help="package file output by make_pkg")
    add ('-f','--fluxcal', help='Fluxcal solution file (output of make_fluxcal.py)', dest='fx')
    add ('-O', '--outdir', help='Output directory', default="./", dest='odir')
    add ('-v','--verbose', action='store_true', dest='v')
    return agp.parse_args ()

if __name__ == "__main__":
    args    = get_args ()
    ####################################
    bn      = os.path.basename ( args.pkg )
    bnf,_   = os.path.splitext ( bn )
    odir    = args.odir
    sofile  = os.path.join ( args.odir, bnf + ".fluxcal.npz" )
    ####################################
    # fig     = plt.figure ()
    ## read file
    pkg     = np.load ( args.pkg )
    obstime = pkg['obstime']
    ####################################
    ## read fluxcal solution
    fluxcal   = np.load ( args.fx )
    fxmjd     = fluxcal['mjd']
    sefd      = np.ma.MaskedArray ( fluxcal['data_sefd_jy'], mask=fluxcal['mask_sefd_jy'] )
    sefd      = sefd.reshape ((-1,1))
    #########
    ## time difference
    dt        = abs ( obstime - fxmjd )
    if dt > 1.0:
        raise RuntimeError (f" Fluxcal solution and burst are far apart in time!")
    ####################################
    # read data
    data    = pkg['data'][0]
    wts     = np.ones (pkg['data'].shape, dtype=bool)
    ww      = np.array (pkg['wts'], dtype=bool)
    wts[:,:,ww,:] = False
    ww      = wts[0]
    mata    = np.ma.array (data, mask=ww, fill_value=np.nan)


    ## read meta
    Nch     = int ( pkg['nchan'] )
    Nbin    = pkg['nbin']

    on_mask = np.zeros ( pkg['nbin'], dtype=bool )
    # ff_mask = np.zeros ( pkg['nchan'], dtype=bool )

    ## 20230314 : everything that is not ON is OFF
    ons     = slice ( pkg['tstart'], pkg['tstop'] )
    on_mask[ons]   = True

    ofs     = slice ( pkg['fstart'], pkg['fstop'] )
    mata.mask[:,0:pkg['fstart'],:]  = True
    mata.mask[:,pkg['fstop']:,:]    = True
    # ff_mask[ofs]   = True

    nsamp   = mata.shape[2]
    mask    = ww[0].sum (1) == 0.0
    zask    = ww[0].sum (1) != 0.0
    # ff_mask = ff_mask & mask

    # axes
    tsamp   = float (pkg['dur']) / float ( nsamp )
    times   = np.linspace ( 0., float(pkg['dur']), nsamp )
    times   *= 1E3
    freqs     = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen']
    freq_list = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], Nch, endpoint=True) + pkg['fcen']

    freq_lo   = freq_list.min ()
    freq_hi   = freq_list.max ()

    ## Stokes ON pulse
    I       = mata[0]
    # I_on    = np.array ( mata[0,ff_mask][...,on_mask] )
    I_on    = mata[0,...,on_mask]
    ## Stokes OFF pulse
    # I_off   = np.array ( mata[0,ff_mask][...,~on_mask] )
    I_off   = mata[0,...,~on_mask]
    ## freq_list
    # freq_list = freq_list [ ff_mask ]

    ## Sum over ON pulse
    I_sum_on  = np.sum ( I_on, 1 )
    ## Choose high S/N, avoid channels with non-positive I
    I_std_mask= np.std ( I_on, 1 )
    I_off_mean= np.mean (I_off, 1)
    for i,ii in enumerate (I_sum_on):
        if ( ii < 1.66 * I_std_mask[i] ) or ( ii < I_off_mean[i] ):
            I_on.mask[i]   = True
            I_off.mask[i]  = True

    ## sum over time
    ## per channel std-dev
    I_std     = np.std ( I_off, 1 )
    I_on_f    = I_on .mean (0)[:,np.newaxis]
    I_off_f   = I_off.mean (0)[:,np.newaxis]
    # I         = I / I_off_f
    deflection  = I_on_f / I_off_f
    ## if deflection less than 1.0 flag
    for i,d in enumerate ( deflection ):
        if d < 1.0:
            # I.mask[i]          = True
            deflection.mask[i] = True

    nON       = np.sqrt ( ons.stop - ons.start )
    # 20230313 : use whole pulse region to compute the standard deviation
    # 20230313 : and multiply with sqrt ( width )
    I_err     = nON * I_std
    ####################################
    ## apply solution
    """
    idk why i have to apply on I

    i should be applying on deflection - 1.0

    but my I is atleast gain calibrated
    """
    # cal_burst   = I * sefd * ( deflection - 1.0 )
    cal_burst   = sefd *  I
    pp          = cal_burst.mean(0)
    # plt.plot    ( times, pp * 1E3, c='b' )
    # plt.axvline ( times[pkg['tstart']], c='r', ls=':' )
    # plt.axvline ( times[pkg['tstop']], c='r', ls=':' )
    # plt.show ()
    # adf
    ####################################
    ## measure the following
    #### width_ms
    #### peakflux_mjy
    #### fluence_jyms
    #### burst_bw_mhz
    #### burst_fcen_mhz
    ####################################
    ## error propagation
    ## save
    ret                     = dict ()
    # times is in ms
    ret['width_ms']         = times[pkg['tstop']] - times[pkg['tstart']]
    ret['burst_bw_mhz']     = abs ( freqs[pkg['fstop']] - freqs[pkg['fstart']] )
    ret['burst_fcen_mhz']   = ( freqs[pkg['fstop']] + freqs[pkg['fstart']] ) * 0.5
    ## 1E3 because sefd was in Jy
    ret['fluence_jyms']     = np.mean ( pp[ons] * 1E3 )  
    ret['peakflux_mjy']     = np.max ( pp[ons] * 1E3 )
    ## logging
    ret['fluxcal_file']     = os.path.basename ( args.fx )
    ret['burst_file']       = bn
    ret['solution_mjd']     = fxmjd
    ret['burst_mjd']        = obstime
    ##
    np.savez ( sofile, **ret )


