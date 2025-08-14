"""
measure the ionospheric RM at the time of input
using spinifex
"""
# coding: utf-8
import os
import numpy as np
import pandas as pd

import astropy.time as at
import astropy.coordinates as asc
import astropy.units as au
import astropy.io.fits as aif

from spinifex import get_rm

SPINIFEX_DIR = "/tmp/temp_spinifex"
################################################
gmrt = asc.EarthLocation (
        lat='19d06m',
        lon='74d03m',
        height=300, 
)

RMS   = {"0329+54":-64.33, "0139+5814":-94.13, "R3":np.nan}

def get_parallactic_angle ( sc, tobs ):
    """ source coordinates, mjd --> parallactic angle (degree) """

    lst   = tobs.sidereal_time ( 'mean', longitude=gmrt.lon, model=None )
    h     = (lst - sc.ra).radian
    q     = np.arctan2 ( 
            np.sin ( h ), 
            np.tan ( gmrt.lat.radian ) * np.cos ( sc.dec.radian ) - 
            np.sin ( sc.dec.radian ) * np.cos ( h )
    )
    return q

def get_ionospheric_rm ( sc, tobs ):
    """
    wrapper
    """
    rm  = get_rm.get_rm_from_skycoord ( loc=gmrt, times=tobs, source=sc, output_directory=SPINIFEX_DIR, prefix="uqr", server='cddis' )
    return rm

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
    ####################
    tobs      = at.Time ( mjd, format='mjd' )
    sc        = asc.SkyCoord ( ra, dec, unit=(au.hourangle, au.degree)  )
    #### 
    taxis     = tobs + ( np.array ([-60, -30, 0, 30, 60]) * au.min )
    ## get PAL
    pal       = get_parallactic_angle ( sc, tobs ) * au.radian
    irm       = get_ionospheric_rm  ( sc, taxis )
    ionos_rm = irm.rm[2]
    print (f" {ar}")
    print (f"\tSource={src_name} at MJD={mjd:.6f} parallactic_angle={pal.to(au.degree):.2f}")
    print (f"\tIonospheric RM = {ionos_rm:.3f}")
    if src_name in RMS.keys():
        corr_rm   = ionos_rm + RMS[src_name]
        print (f"\tCorrection RM = {corr_rm:.3f}")

##############################
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




