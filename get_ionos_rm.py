"""
measure the ionospheric RM at the time of input
"""
# coding: utf-8
import os
import numpy as np
import pandas as pd

import astroplan as ap

import astropy.time as at
import astropy.coordinates as asc
import astropy.units as au
import astropy.io.fits as aif
################################################
gmrt = asc.EarthLocation (
        lat='19d06m',
        lon='74d03m',
        height=300, 
)
observer = ap.Observer (location=gmrt)

RMS   = {"0329+54":-64.33, "0139+5814":-94.13}

def get_parallactic_angle ( ra, dec, mjd ):
    """ source coordinates, mjd --> parallactic angle (degree) """
    atm  = at.Time ( mjd, format='mjd' )
    atc  = asc.SkyCoord ( ra, dec, unit=(au.hourangle, au.degree)  )
    pal  = observer.parallactic_angle (atm, atc).to(au.degree).value
    return pal

def get_ionospheric_rm ( ra, dec, mjd, duration=30., nsteps=2, prefix='codg' ):
    """ uses RMextract """
    from RMextract.getRM import getRM
    ###
    pointing = []
    if isinstance(ra, str) and isinstance(dec, str):
        atc      = asc.SkyCoord ( ra, dec, unit=(au.hourangle, au.degree)  )
        pointing = [ atc.ra.radian, atc.dec.radian ]
    elif isinstance ( ra, float ) and isinstance ( dec, float ):
        pointing = [ ra, dec ]
    else:
        raise ValueError("pointing error")
    gmrt_pos = [ 1656342.30, 5797947.77, 2073243.16 ]
    time     = mjd * 86400.0
    ###
    ret      = getRM ( 
        radec=pointing,
        stat_positions=[gmrt_pos],
        timestep=nsteps,
        timerange=[time,time+duration],
        prefix=prefix,
        # server="http://cddis.gsfc.nasa.gov"
        server="ftp://gssc.esa.int/gnss/products/ionex/"
    )
    ###
    retrm    = ret['RM']['st1'].mean()
    return retrm

def action ( ar ):
    cal           = aif.open ( ar )
    names = [fi.name for fi in cal]
    if args.v:
        print (f" From FITS received tables = {names}")
    ## get source name
    ptab  = cal[0]
    ## get MJD
    src_name  = ptab.header['SRC_NAME']
    ra        = ptab.header['RA']
    dec       = ptab.header['DEC']
    stt_imjd  = ptab.header['STT_IMJD']
    stt_smjd  = ptab.header['STT_SMJD']
    stt_offs  = ptab.header['STT_OFFS']
    mjd       = ( stt_imjd ) + ( stt_smjd / 86400. ) + ( stt_offs / 86400 )
    if args.v:
        print (f" RA/DEC = {ra},{dec}")
    ## get PAL
    pal_deg   = get_parallactic_angle ( ra, dec, mjd )
    ## get iRM
    # ionos_rm  = get_ionospheric_rm ( ra, dec, mjd, prefix='upcg' )
    ionos_rm  = get_ionospheric_rm ( ra, dec, mjd, prefix='uqrg' )
    # ionos_rm  = get_ionospheric_rm ( ra, dec, mjd, prefix='cgim' )
    print (f" {ar}")
    print (f"\tSource={src_name} at MJD={mjd:.6f} parallactic_angle={pal_deg:.2f} deg")
    print (f"\tIonospheric RM = {ionos_rm:.3f}")
    corr_rm   = ionos_rm + RMS[src_name]
    print (f"\tCorrection RM = {corr_rm:.3f}")

##############################
SOL = 'pacv_sols/3C48_NGON_bm3_pa_550_200_32_16mar2021.raw.noise.ar.pazi.pacv'
def get_args ():
    import argparse as agp
    ap   = agp.ArgumentParser ('get_ionos_rm', description='Measures ionospheric RM contribution', epilog='Part of GMRT/FRB')
    add  = ap.add_argument
    add ('ar', help='archive file', nargs='+')
    add ('-v', '--verbose', help='Verbose', dest='v', action='store_true')
    return ap.parse_args ()

################################################
if __name__ == "__main__":
    args          = get_args ()
    ###
    # read file
    for a in args.ar:
        action ( a )



