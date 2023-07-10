"""
make_fluxcal

measures deflection from a :make_np.py: file
and writes solution in a json file 

the solution is then applied to bursts
using :apply_fluxcal.py:
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

# from my_pacv import read_pkl
def read_pkl (file):
    """ return dict(fbw, nchan, fcen, mjd, source), freq, sc """
    import pickle as pkl
    with open (file, "rb") as f:
        k  = pkl.load ( f, encoding='latin1' )
    sc = np.ma.array (k['data'][0], mask=k['wts'][0], fill_value=np.nan)
    f  = np.linspace ( -0.5 * k['fbw'], 0.5 * k['fbw'], k['nchan'] ) + k['fcen']
    wt = np.array (k['wts'][0].sum ( (0,2) ), dtype=bool)
    freq = np.ma.array ( f, mask=wt, fill_value=np.nan )
    return dict(fbw=k['fbw'], fcen=k['fcen'], mjd=k['mjd'], source=k['src'], nchan=k['nchan'],basis=k['basis']), freq, sc

#############################################
"""
calibrator spectra polynomial coefficients from Perley+Butler (2017)
"""
CALS              = dict ()
CALS['3C147']     = [1.4516, -0.6961, -0.2007, 0.0640, -0.0464, 0.0289]
CALS['3C138']     = [1.0088, -0.4981, -0.1552, -0.0102, 0.0223]
CALS['3C48']      = [1.3253, -0.7553, -0.1914, 0.0498]

#############################################
def get_args ():
    import argparse
    agp = argparse.ArgumentParser ("make_fluxcal", description="Measures SEFD using calibration solution file", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add ('pkl', help='Pickle file (output of make_np.py)',)
    add ('-O', '--outdir', help='Output directory', default="./", dest='odir')
    add ('-v','--verbose', action='store_true', dest='v')
    return agp.parse_args ()

def get_cal_flux (source, freq):
    """
    source: source name
    freq: frequency list

    on: CAL ON
    off: CAL OFF

    source-flux = 
    (
        ( ( source.on / source.off) - 1) / 
        ( ( cal.on    / cal.off)    - 1)
    ) * cal.flux

    since source is not provided here as input, 
    this just calculates 
    > cal.flux / ( ( cal.on / cal.off ) - 1 )

    but in this just return cal.flux
    """
    src      = source.upper ()
    ## get flux from Perley+Butler model
    if src not in CALS.keys():
        print (" received calibrator = {sl}".format(sl=sl))
        raise RuntimeError (" Calibrator not found in internal database")
    coeff      = CALS[src]
    ## log frequency in GHz
    l_f_g      = np.log10 (freq / 1E3)
    pp         = np.poly1d (list(reversed(coeff)))
    cal_flux   = np.power (10, pp (l_f_g))
    return cal_flux

if __name__ == "__main__":
    args = get_args ()
    #######################
    FILE_UNCAL  = args.pkl
    base,_      = os.path.splitext ( os.path.basename ( FILE_UNCAL ) )
    sofile      = os.path.join ( args.odir, base + ".fluxcal.sol.npz" )
    pnfile      = os.path.join ( args.odir, base + ".fluxcal.png" )
    #######################
    ########### read
    pkg, freq, sc  = read_pkl ( FILE_UNCAL )
    nchan       = freq.size
    source      = pkg['source']
    mjd         = pkg['mjd']
    ## ON phase solve
    ## ON-phase is more than 60% of the maximum
    pp          = sc[0].mean(0)
    mask        = pp >= (0.60 * pp.max())
    ## here i need to mean over axis=0???
    i_on        = sc[0, :, mask].mean(0)
    i_of        = sc[0, :, ~mask].mean(0)
    # off_std     = sc[...,~mask].std(-1)
    ## deflection
    deflection  = i_on / i_of
    ## flag everything where deflection is <1.0
    for i,d in enumerate ( deflection ):
        if d < 1.0:
            deflection.mask[i] = True
            freq.mask[i]       = True

    if np.sum( freq.mask ) > (nchan*0.5):
        raise RuntimeError(" Almost all channels have been masked!!")

    ## cal flux
    cal_flux    = get_cal_flux ( source, freq )
    ## sefd
    sefd        = cal_flux / ( deflection - 1.0 )
    mean_sefd   = np.mean ( sefd )
    err_sefd    = np.std ( sefd )
    ##########################################
    ## save file
    RET            = dict()
    RET['source']  = source
    RET['data_sefd_jy']    = sefd.data
    RET['mask_sefd_jy']    = sefd.mask
    RET['mean_sefd_jy']    = mean_sefd
    RET['std_sefd_jy']    = err_sefd 
    RET['mjd']     = mjd
    RET['freq_mhz']    = freq
    np.savez (sofile, **RET)
    ##########################################
    
    fig                    = plt.figure (dpi=300, figsize=(7,5))
    xsource,xdefl,xsefd    = fig.subplots ( 3,1,sharex=True )

    xsource.plot ( freq, cal_flux, c='b' )
    xsource.set_ylabel ("Flux / Jy")

    xdefl.plot ( freq, deflection, c='r' )
    xdefl.axhline (1.0, ls=':',c='k', alpha=0.6)
    xdefl.set_ylabel ('Deflection')
    xdefl.set_yscale ('log')

    xsefd.plot ( freq,  1000.0*sefd, c='g')
    xsefd.set_ylabel ('SEFD / mJy')

    xsefd.set_xlabel ('Freq / MHz')

    fig.suptitle (f"SEFD = {mean_sefd*1E3:.3f} +- {err_sefd*1E3:.3f} mJy")

    fig.savefig (pnfile, dpi=300, bbox_inches='tight')
    # plt.show ()
