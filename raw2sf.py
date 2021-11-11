"""
Convert Filterbank files to PSRFITS file

Modified to convert uGMRT raw filterbank into PSRFITS snippets

Major change(s):
    Only support for full stokes data
    Output PSRFITS is search-mode
    `nbits` is 8 since uGMRT raw is 16bit

Original Source: https://github.com/rwharton/fil2psrfits
Copied and modified from: https://github.com/thepetabyteproject/your/blob/master/your/formats/fitswriter.py

XXX everything is done in memory because of dependency on `astropy.io.fits`
using `fitsio` would alleviate and help run this more smoothly and on large size scans

TODO
 - proper logging?
 - RFI excision?

"""
import os

import tqdm

import logging

import numpy as np

from astropy.io import fits

from obsinfo import *

from redigitize import Redigitize

################################
logger = logging.getLogger (__name__)

def get_args ():
    import argparse
    agp = argparse.ArgumentParser ("raw2sf", description="Converts uGMRT raw to search-mode PSRFITS", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add ('-c,--nchan', help='Number of channels', type=int, required=True, dest='nchans')
    add ('-b,--bit-shift', help='Bitshift', type=int, dest='bitshift', default=6)
    add ('--lsb', help='Lower subband', action='store_true', dest='lsb')
    add ('--usb', help='Upper subband', action='store_true', dest='usb')
    add ('--gulp', help='Samples in a block', dest='gulp', default=2048, type=int)
    add ('--beam-size', help='Beam size in arcsec', dest='beam_size', default=4, type=float)
    add ('-s', '--source', help='Source', choices=MISC_SOURCES, required=True)
    add ('-O', '--outdir', help='Output directory', default="./")
    add ('-d','--debug', action='store_const', const=logging.DEBUG, dest='loglevel')
    add ('-v','--verbose', action='store_const', const=logging.INFO, dest='loglevel')
    add ('raw', help='Path to raw file',)
    return agp.parse_args ()

if __name__ == "__main__":
    args = get_args ()
    logging.basicConfig (level=args.loglevel, format="%(asctime)s %(levelname)s %(message)s")
    #################################
    # subband logic
    if args.lsb and args.usb:
        raise ValueError (" cannot specify both subbands ")
    if not (args.lsb or args.usb):
        raise ValueError (" must specify atleast one subband ")
    #################################
    GULP = args.gulp
    nch  = args.nchans
    npl  = 4
    ###
    raw  = args.raw
    baw  = os.path.basename (raw)
    hdr  = raw + ".hdr"
    logging.info (f"Raw file           = {raw}")
    logging.info (f"Raw header file    = {hdr}")
    ### read time
    rawt = read_hdr (hdr)
    logging.info (f"Raw MJD            = {rawt.mjd:.5f}")
    ### read raw
    rfb  = np.memmap (raw, dtype=np.int16, mode='r', offset=0, )
    fb   = rfb.reshape ((-1, nch, npl))
    logging.debug (f"Raw shape         = {fb.shape}")
    ### read freq/tsamp
    band = get_band  (baw)
    tsamp= get_tsamp (band, nch)
    freqs= get_freqs (band, nch, lsb=args.lsb, usb=args.usb)
    logging.info (f"Raw band           = {band}")
    logging.debug (f"Tsamp             = {tsamp}")
    logging.debug (f"Frequencies       = {freqs[0]:.3f} ... {freqs[-1]:.3f}")
    #################################
    rdi        = Redigitize (GULP, nch, npl)
    #################################
    nsamples   = fb.shape[0]
    nrows      = nsamples // GULP
    #print ("################################")
    #nrows      = 4
    #print ("################################")
    fsamples   = nrows * GULP
    last_row   = nsamples - fsamples
    logging.debug (f"Total samples     = {nsamples:d}")
    logging.debug (f"Number of rows    = {nrows:d}")
    logging.debug (f"Samples in lastrow= {last_row:d}")

    row_size   = GULP * nch * npl * 2

    tr         = tqdm.tqdm (range (0, fsamples, GULP), desc='raw2sf', unit='blk')
    #################################
    ## setup psrfits file
    outfile = os.path.join (args.outdir, baw + ".sf")
    logging.info (f"Output search-mode psrfits = {outfile}")

    # Fill in the ObsInfo class
    d = BaseObsInfo (rawt.mjd, 'search')
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
    # udat     = np.zeros ((GULP, nch, npl), dtype=np.int16)
    # pdat     = np.zeros ((nch, npl, GULP), dtype=np.uint16)
    for i in tr:
        #udat[:] = 0
        #pdat[:] = 0
        pkg    = fb[i:(i+GULP)]
        ### data wrangling
        rdi (pkg)
        """
        data ordering : (nchans, npol, nsblk*nbits/8)

        DATA is already in nbit=16
        
        pkg shape = (nsamps, nchans, npol)
        dat shape = (nchans, npol  , nsamps)
        """
        ###
        ## GMRT pol order to full stokes IQUV
        ## may need to complicate to support total intensity
        # udat[...,0] = pkg[...,0] + pkg[...,2]
        # udat[...,1] = pkg[...,1]
        # udat[...,2] = pkg[...,3]
        # udat[...,3] = pkg[...,0] - pkg[...,2]

        ###
        ## axis ordering
        # pdat[:] = np.moveaxis (udat, 0, -1)

        # subint_sf.data[isubint]['DATA'] = np.uint8 (pdat.T[:] >> args.bitshift)
        subint_sf.data[isubint]['DATA'][:]      = rdi.dat[:]
        subint_sf.data[isubint]['DAT_SCL'][:]   = rdi.dat_scl.ravel()
        subint_sf.data[isubint]['DAT_OFFS'][:]  = rdi.dat_offs.ravel()
        isubint = isubint + 1

        ## flush?
        write_sf.flush ()

    ##
    write_sf.close ()

