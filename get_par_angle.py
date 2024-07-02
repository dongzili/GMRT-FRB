"""
get parallactic_angle 
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
gmrt  = asc.EarthLocation.from_geocentric (1656342.30, 5797947.77, 2073243.16, unit="m")
observer = ap.Observer (location=gmrt)

R3_sc = asc.SkyCoord ( 29.50312583 * au.degree, 65.71675422 * au.degree, frame='icrs' )

def get_parallactic_angle ( ra, dec, mjd ):
    """ source coordinates, mjd --> parallactic angle (degree) """
    atm  = at.Time ( mjd, format='mjd' )
    atc  = asc.SkyCoord ( ra, dec, unit=(au.hourangle, au.degree)  )
    print ( atc.ra.hms, atc.dec.dms )
    pal  = observer.parallactic_angle (atm, atc).to(au.degree).value
    return pal

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
    # print (f" {ar}")
    # print (f"\tSource={src_name} at MJD={mjd:.6f} parallactic_angle={pal_deg:.2f} deg")
    print ( f"{ar},{mjd:.8f},{pal_deg:.4f}" )

##############################
SOL = 'pacv_sols/3C48_NGON_bm3_pa_550_200_32_16mar2021.raw.noise.ar.pazi.pacv'
def get_args ():
    import argparse as agp
    ap   = agp.ArgumentParser ('get_par_angle', description='Measures parallactic_angle', epilog='Part of GMRT/FRB')
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



