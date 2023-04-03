"""
Convert Filterbank files to PSRFITS file

`zips` noise diode ON and OFF scans into a single PSRFITS snippet scan

Major change(s):
    Only support for full stokes data
    Output PSRFITS is search-mode
    `nbits` is 32 since uGMRT raw is 16bit

Original Source: https://github.com/rwharton/fil2psrfits
Copied and modified from: https://github.com/thepetabyteproject/your/blob/master/your/formats/fitswriter.py

TODO
 - proper logging?
 - RFI excision?

"""
import os

import tqdm

import logging

import numpy as np

import datetime as dt
from   dateutil import tz

import astropy.time  as at
import astropy.units as au
import astropy.coordinates as asc

from astropy.io import fits

from obsinfo import *

from redigitize import Redigitize

logger = logging.getLogger (__name__)
###################################################
def get_args ():
    import argparse
    agp = argparse.ArgumentParser ("noise2sf", description="Zips uGMRT noise diode raw to search-mode PSRFITS", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add ('-c,--nchan', help='Number of channels', type=int, required=True, dest='nchans')
    add ('-b,--sb', help='Sideband',choices=['l','u','lower','upper'], dest='sideband', required=True)
    add ('-f,--feed', help='Feed',choices=['cir','lin','c','l'], dest='feed', required=True)
    add ('--gulp', help='Samples in a block', dest='gulp', default=2048, type=int)
    #add ('-b,--bandwidth', help='Bandwidth', dest='bw', required=True, type=float, choices=[100,200,400])
    #add ('-f,--freq', help='Frequency edge', dest='fedge', required=True, type=float, choices=[550, 1460, 1260, 1200])
    #add ('-i,--fftint', help='FFT integration', dest='fftint', required=True, type=int, choices=[8,16,32])
    add ('--beam-size', help='Beam size in arcsec', dest='beam_size', default=4, type=float)
    add ('-s', '--source', help='Source', choices=SOURCES, required=True)
    add ('-O', '--outdir', help='Output directory', default="./")
    add ('-d','--debug', action='store_const', const=logging.DEBUG, dest='loglevel')
    add ('-v','--verbose', action='store_const', const=logging.INFO, dest='loglevel')
    add ('raw_on', help='Path to ON raw file',)
    add ('raw_off', help='Path to OFF raw file',)
    return agp.parse_args ()

if __name__ == "__main__":
    args = get_args ()
    logging.basicConfig (level=args.loglevel, format="%(asctime)s %(levelname)s %(message)s")
    #################################
    if not os.path.exists (args.outdir):
        os.mkdir (args.outdir)
    #################################
    # subband logic
    LSB     = False
    USB     = False
    if args.sideband == 'l' or args.sideband == 'lower':
        LSB = True
    else:
        USB = True
    # feed logic
    feed    = ''
    CIRC    = False
    LIN     = False
    if args.feed == 'c' or args.feed == 'cir':
        CIRC = True
        feed = 'CIRC'
    else:
        LIN  = True
        feed = 'LIN'
    #################################
    GULP = args.gulp
    hGULP= GULP // 2
    nch  = args.nchans
    npl  = 4
    ############
    raw_on  = args.raw_on
    baw_on  = os.path.basename (raw_on)
    hdr_on  = raw_on + ".hdr"
    raw_of  = args.raw_off
    baw_of  = os.path.basename (raw_of)
    hdr_of  = raw_of + ".hdr"
    logging.info (f"Raw ON file            = {raw_on}")
    logging.info (f"Raw ON header file     = {hdr_on}")
    logging.info (f"Raw OFF file           = {raw_of}")
    logging.info (f"Raw OFF header file    = {hdr_of}")
    #### band check
    """
    this is skipped because now the input is given form CL and not parsed form the file name
    """
    band_on = get_band (baw_on)
    band_of = get_band (baw_of)
    if band_on['fftint'] != band_of['fftint']:
        raise ValueError ("FFT integration not same")
    if band_on['bw'] != band_of['bw']:
        raise ValueError ("Bandwidth not same")
    if band_on['fedge'] != band_of['fedge']:
        raise ValueError ("Band not same")
    #band_on = dict (beam='on', fedge=args.fedge, bw=args.bw, fftint=args.fftint)
    #band_of = dict (beam='of', fedge=args.fedge, bw=args.bw, fftint=args.fftint)
    ### read time
    rawt_on = read_hdr (hdr_on)
    rawt_of = read_hdr (hdr_of)
    #logging.warning (f" setting random MJD")
    #rawt_on  = at.Time (59243.00, format='mjd')
    #rawt_of  = at.Time (59243.01, format='mjd')
    logging.info (f"Raw ON MJD            = {rawt_on.mjd:.5f}")
    logging.info (f"Raw OFF MJD           = {rawt_of.mjd:.5f}")
    ### read raw
    rfb_on  = np.memmap (raw_on, dtype=np.int16, mode='r', offset=0, )
    fb_on   = rfb_on.reshape ((-1, nch, npl))
    rfb_of  = np.memmap (raw_of, dtype=np.int16, mode='r', offset=0, )
    fb_of   = rfb_of.reshape ((-1, nch, npl))
    logging.debug (f"Raw ON shape         = {fb_on.shape}")
    logging.debug (f"Raw OFF shape        = {fb_of.shape}")
    ### read freq/tsamp
    band = band_on
    tsamp= get_tsamp (band, nch)
    freqs= get_freqs (band, nch, lsb=LSB, usb=USB)
    logging.debug (f"Frequencies       = {freqs[0]:.3f} ... {freqs[-1]:.3f}")
    #################################
    rdi        = Redigitize (GULP, nch, npl, feed)
    #################################
    nsamples   = min (fb_on.shape[0],fb_of.shape[0])
    nrows      = nsamples // GULP
    #print ("################################")
    #nrows      = 4
    #print ("################################")
    fsamples   = nrows * hGULP
    last_row   = nsamples - fsamples
    logging.debug (f"Total samples     = {nsamples:d}")
    logging.debug (f"Number of rows    = {nrows:d}")
    logging.debug (f"Samples in lastrow= {last_row:d}")

    row_size   = GULP * nch * npl * 2

    tr         = tqdm.tqdm (range (0, fsamples, hGULP), desc='noise2sf', unit='blk')
    #################################
    ## setup psrfits file
    outfile = os.path.join (args.outdir, baw_on + ".noise.sf")
    logging.info (f"Output search-mode psrfits = {outfile}")

    # Fill in the ObsInfo class
    d = BaseObsInfo (rawt_of.mjd, 'cal', circular=CIRC, linear=LIN)
    d.fill_freq_info (nch, band['bw'], freqs)
    d.fill_source_info (args.source, RAD[args.source], DECD[args.source])
    d.fill_beam_info (args.beam_size)
    d.fill_data_info (tsamp) 

    # subint columns
    t_row     = GULP * tsamp
    ## XXX not considering the half row

    tsubint   = np.ones (nrows, dtype=np.float64) * t_row
    offs_sub  = (np.arange(nrows) + 0.5) * t_row
    lst_sub   = d.get_lst_sub (offs_sub)

    ra_deg, dec_deg   = d.sc.ra.degree, d.sc.dec.degree
    scg               = d.sc.galactic
    l_deg, b_deg      = scg.l.value, scg.b.value
    ra_sub            = np.ones(nrows, dtype=np.float64) * ra_deg
    dec_sub           = np.ones(nrows, dtype=np.float64) * dec_deg
    glon_sub          = np.ones(nrows, dtype=np.float64) * l_deg
    glat_sub          = np.ones(nrows, dtype=np.float64) * b_deg
    fd_ang            = np.zeros(nrows, dtype=np.float32)
    pos_ang           = np.zeros(nrows, dtype=np.float32)
    par_ang           = np.zeros(nrows, dtype=np.float32)
    tel_az            = np.zeros(nrows, dtype=np.float32)
    tel_zen           = np.zeros(nrows, dtype=np.float32)
    dat_freq          = np.vstack([freqs] * nrows).astype(np.float32)
    dat_wts           = np.ones((nrows, nch), dtype=np.float32)
    dat_offs          = np.zeros((nrows,nch,npl),dtype=np.float32)
    dat_scl           = np.ones((nrows,nch,npl),dtype=np.float32)
    # dat               = np.zeros((n_subints, row_size), dtype=np.uint8)
    dat               = np.array ([], dtype=np.uint8)

    # Fill in the headers
    phdr      = d.fill_primary_header(chan_dm=0., scan_len=t_row * nrows)
    subinthdr = d.fill_search_table_header(GULP)
    fits_data = fits.HDUList()

    subint_columns = [
        fits.Column(name="TSUBINT",  format="1D", unit="s", array=tsubint),
        fits.Column(name="OFFS_SUB", format="1D", unit="s", array=offs_sub),
        fits.Column(name="LST_SUB",  format="1D", unit="s", array=lst_sub),
        fits.Column(name="RA_SUB",   format="1D", unit="deg", array=ra_sub),
        fits.Column(name="DEC_SUB",  format="1D", unit="deg", array=dec_sub),
        fits.Column(name="GLON_SUB", format="1D", unit="deg", array=glon_sub),
        fits.Column(name="GLAT_SUB", format="1D", unit="deg", array=glat_sub),
        fits.Column(name="FD_ANG",   format="1E", unit="deg", array=fd_ang),
        fits.Column(name="POS_ANG",  format="1E", unit="deg", array=pos_ang),
        fits.Column(name="PAR_ANG",  format="1E", unit="deg", array=par_ang),
        fits.Column(name="TEL_AZ",   format="1E", unit="deg", array=tel_az),
        fits.Column(name="TEL_ZEN",  format="1E", unit="deg", array=tel_zen),
        fits.Column(name="DAT_FREQ", format=f"{nch:d}E", unit="MHz", array=dat_freq),
        fits.Column(name="DAT_WTS",  format=f"{nch:d}E", array=dat_wts),
        fits.Column(name="DAT_OFFS", format=f"{nch*npl:d}E", array=dat_offs),
        fits.Column(name="DAT_SCL",  format=f"{nch*npl:d}E", array=dat_scl),
        fits.Column(
            name="DATA",
            format=f"{nch*npl*GULP:d}B",
            dim=f"({nch}, {npl}, {GULP})",
            array=dat,
        ),
    ]

    # Add the columns to the table
    subint_table = fits.BinTableHDU(
        fits.FITS_rec.from_columns(subint_columns), name="subint", header=subinthdr
    )

    # Add primary header
    primary_hdu = fits.PrimaryHDU(header=phdr)

    fits_data.append(primary_hdu)
    fits_data.append(subint_table)
    fits_data.writeto(outfile, overwrite=True)
    logging.debug ('psrfits file setup done')

    logging.debug ('reopening psrfits')
    write_sf     = fits.open (outfile, mode='update')
    subint_sf    = write_sf[1]

    #sys.exit (0)
    #################################
    ## work loop
    isubint = 0
    rdat     = np.zeros ((GULP, nch, npl), dtype=np.int16)
    sdat     = np.zeros ((GULP, nch, npl), dtype=np.int16)
    # slices

    for i in tr:
        rdat[:] = 0
        sdat[:] = 0
        ### reading
        pkg_on  = fb_on[i:(i+hGULP)]
        pkg_of  = fb_of[i:(i+hGULP)]
        ### data wrangling
        """
        data ordering : (nchans, npol, nsblk*nbits/8)

        DATA is already in nbit=16
        
        pkg shape = (nsamps, nchans, npol)
        dat shape = (nchans, npol  , nsamps)
        """
        ###
        ## OFF-ON-OFF
        rdat[:hGULP] = pkg_on[:]
        rdat[hGULP:] = pkg_of[:]
        sdat[:]     = np.roll (rdat, 768, axis=0)
        #sdat[:]     = np.roll (rdat, 0, axis=0)

        ###
        ## data wrangling
        rdi (sdat)

        ###
        ## axis ordering
        # pdat[:] = np.moveaxis (sdat, 0, -1)

        # subint_sf.data[isubint]['DATA'] = np.uint8 (pdat.T[:] >> args.bitshift)
        subint_sf.data[isubint]['DATA'][:]      = rdi.dat[:]
        subint_sf.data[isubint]['DAT_SCL'][:]   = rdi.dat_scl.ravel()
        subint_sf.data[isubint]['DAT_OFFS'][:]  = rdi.dat_offs.ravel()
        isubint = isubint + 1

        ## flush?
        write_sf.flush ()

    ##
    write_sf.close ()

