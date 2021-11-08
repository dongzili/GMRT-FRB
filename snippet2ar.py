"""
Convert Filterbank files to PSRFITS file

Modified to convert uGMRT raw filterbank into PSRFITS snippets

Major change(s):
    Output PSRFITS is fold-mode which covers a burst snippet.
    Motivation for doing so is to treat burst as an "integrated pulse profile"
    In fold-mode the `nbits` is 16bit signed integers (post scale, offset removal)

Original Source: https://github.com/rwharton/fil2psrfits
Copied and modified from: https://github.com/thepetabyteproject/your/blob/master/your/formats/fitswriter.py

TODO
 - proper logging?
 - RFI excision?


"""
import os

import logging

import numpy as np

from astropy.io import fits

from obsinfo import *

logger = logging.getLogger (__name__)

ARFILE="{tmjd:15.10f}_lof{freq:3.0f}_{source}.ar"
###################################################
DM_CONST = 4.148741601E3
def dispdelay(DM,LOFREQ,HIFREQ):
    return DM * DM_CONST * (LOFREQ**-2 - HIFREQ**-2) 

def dedisperser (IN, freq_delays):
    """assumes de-dispersion is valid
    """
    ishape = IN.shape
    max_delay = freq_delays[0]

    osamps = ishape[0]- max_delay
    oshape = (osamps, *ishape[1:])

    if osamps <= 0:
        raise ValueError ("incompatible de-dispersion")

    OUT = np.zeros (oshape, dtype=IN.dtype)
    u,v = 0,0
    for ifreq, idelay in enumerate (freq_delays):
        """ low to high frequency ordering"""
        u = idelay
        v = u + osamps
        OUT[:,ifreq,...] = IN[u:v,ifreq,...]
    return OUT
###################################################

def get_args ():
    import argparse
    agp = argparse.ArgumentParser ("2ar", description="Converts burst snippets to PSRFITS", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add ('file',help='Raw file')
    add ('-c,--nchan', help='Number of channels', type=int, required=True, dest='nchans')
    add ('-s', '--source', help='Source', choices=SNIP_SOURCES, required=True)
    add ('--lsb', help='Lower subband', action='store_true', dest='lsb')
    add ('--usb', help='Upper subband', action='store_true', dest='usb')
    ###
    add ('--toa',help='TOA csv', required=True, dest='toa')
    ###
    add ('-n,--nbins', help='Samples in the snippet', dest='nbins', default=1024, type=int)
    add ('--beam-size', help='Beam size in arcsec', dest='beam_size', default=4, type=float)
    add ('-v','--verbose', action='store_const', const=logging.INFO, dest='loglevel')
    add ('-d', '--debug', action='store_const', const=logging.DEBUG, dest='loglevel')
    add ('-O', '--outdir', help='Output directory', default="./")
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
    nbin = args.nbins
    nch  = args.nchans
    npl  = 4
    ###
    raw  = args.raw
    baw  = os.path.basename (raw)
    hdr  = raw + ".hdr"
    logging.info (f"Raw file           = {raw}")
    logging.info (f"Raw header file    = {hdr}")
    logging.info (f"Raw band           = {band}")
    ### read time
    rawt = read_hdr (hdr)
    logging.info (f"Raw MJD            = {rawt.mjd:.5f}")
    ### read raw
    rfb  = np.memmap (raw, dtype=np.uint16, mode='r', offset=0, )
    fb   = rfb.reshape ((-1, nch, npl))
    logging.debug (f"Raw shape         = {fb.shape}")
    ### read freq/tsamp
    band = get_band  (baw)
    tsamp= get_tsamp (band, nch)
    freqs= get_freqs (band, nch, lsb=args.lsb, usb=args.usb)
    logging.debug (f"Tsamp             = {tsamp}")
    logging.debug (f"Frequencies       = {freqs[0]:.0f} ... {freqs[-1]:.0f}")
    scan_time = tsamp * fb.shape[0]
    logging.debug (f"Scan len          = {scan_time:.3f} s")
    #################################
    dm   = DMD[args.source]
    logging.debug (f"DM                = {dm} pc/cc")
    # delays
    ref_freq = np.max (freqs) + 0.5*abs (freqs[1]-freqs[0])
    logging.debug (f"Reference freq    = {ref_freq} MHz")
    f_delays = np.zeros (nchans, dtype=np.int64)
    for ichan, freq in enumerate (freqs):
        f_delays[ichan] = int (dispdelay (DM, freq, ref_freq)/tsamp)
    max_delay  = np.max (f_delays)
    logging.debug (f"Delays            = {f_delays[0]:d} ... {f_delays[-1]:d}")
    #################################
    ## read toa csv
    toa  = pd.read_csv (args.toa)
    logging.debug ("TOA file           = {args.toa}")
    it   = tqdm.tqdm (toa.index, unit='toa', desc='Snippet')
    #################################
    nbins = args.nbins
    hbins = nbins // 2
    ## loop
    for i in it:
        sn   = toa.sn[i]
        bmj  = toa.toa[i]
        pt   = at.Time (bmj, format='mjd') - rawt

        ## in file check
        pt_s = pt.to (au.second).value
        if pt_s < 0 or pt_s > scan_time:
            logging.warning (f" burst not in scan index={i:d} S/N={sn:.1f} time={pt_s:.3f} mjd={bmj:.10f}")

        ## output file
        ofile  = os.path.join (args.outdir, ARFILE.format(tmjd=bmj, freq=ref_freq, source=args.source))

        ## slicing
        start_sample = int (pt_s / tsamp) - hbins
        utime_start  = rawt  + (start_sample * tsamp * au.second)
        take_slice   = nbins + max_delay
        ##
        pkg          = fb[start_sample:(start_sample+take_slice)]

        ## dd
        ddpkg        = dedisperser (pkg, f_delays)

        ## stoke-fy
        ddwork       = np.float64 (ddpkg)
        ddwork[ddwork > 2**15] -= 2**16

        stokes_pkg   = np.zeros_like (work_data)
        # IQUV
        stokes_pkg[...,0] = ddwork[...,0] + ddwork[...,2]
        stokes_pkg[...,1] = ddwork[...,1]
        stokes_pkg[...,2] = ddwork[...,3]
        stokes_pkg[...,3] = ddwork[...,0] - ddwork[...,2]

        ## setup ar
        # Fill in the ObsInfo class
        d = BaseObsInfo (utime_start.mjd, 'snippet')
        d.fill_freq_info (nch, band['bw'], freqs)
        d.fill_source_info (args.source, RAD[args.source], DECD[args.source])
        d.fill_beam_info (args.beam_size)
        d.fill_data_info (tsamp, nbins=nbins) 

        ## populate arrays
        ## fold mode, only one burst ==> only one subint
        n_subints = 1

        tstart    = 0.0
        t_subint  = nbins * tsamp
        scan_len  = t_subint * n_subints

        tsubint   = np.ones(n_subints, dtype=np.float64) * t_subint
        offs_sub  = (np.arange(n_subints) + 0.5) * t_subint + tstart
        lst_sub   = d.get_lst_sub (offs_sub)

        ra_deg, dec_deg   = d.sc.ra.degree, d.sc.dec.degree
        scg               = d.sc.galactic
        l_deg, b_deg      = scg.l.value, scg.b.value
        ra_sub            = np.ones(n_subints, dtype=np.float64) * ra_deg
        dec_sub           = np.ones(n_subints, dtype=np.float64) * dec_deg
        glon_sub          = np.ones(n_subints, dtype=np.float64) * l_deg
        glat_sub          = np.ones(n_subints, dtype=np.float64) * b_deg
        fd_ang            = np.zeros(n_subints, dtype=np.float32)
        pos_ang           = np.zeros(n_subints, dtype=np.float32)
        par_ang           = np.zeros(n_subints, dtype=np.float32)
        tel_az            = np.zeros(n_subints, dtype=np.float32)
        tel_zen           = np.zeros(n_subints, dtype=np.float32)
        dat_freq          = np.vstack([freqs] * n_subints).astype(np.float32)

        dat_wts           = np.ones((n_subints, nch), dtype=np.float32)
        dat_offs          = np.zeros((n_subints, nch, npl), dtype=np.float32)
        dat_scl           = np.ones((n_subints, nch, npl), dtype=np.float32)
        # dat               = np.zeros ((n_subints, nbins, nchans, npol), dtype=np.int16)
        dat               = np.zeros ((n_subints, npl, nch, nbins), dtype=np.int16)

        dat[0]            = np.int16 (stokes_pkg.T)

        ## make fold table columns
        # Make the columns
        tbl_columns = [
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
            fits.Column(name="DAT_FREQ", format=f"{nchans:d}E", unit="MHz", array=dat_freq),
            fits.Column(name="DAT_WTS",  format=f"{nchans:d}E", array=dat_wts),
            fits.Column(name="DAT_OFFS", format=f"{nchans*npol:d}E", array=dat_offs),
            fits.Column(name="DAT_SCL",  format=f"{nchans*npol:d}E", array=dat_scl),
            fits.Column(
                name="DATA",
                format=f"{nbins*nchans*npol:d}I",
                dim=f"({nbins}, {nchans}, {npol})",
                array=dat,
            ),
        ]

        # Add the columns to the table
        thdr      = d.fill_fold_table_header ()
        table_hdu = fits.BinTableHDU(
            fits.FITS_rec.from_columns(tbl_columns), name="subint", header=thdr
        )

        # Add primary header
        phdr      = d.fill_primary_header()
        primary_hdu = fits.PrimaryHDU(header=phdr)

        ## polyco table
        polyco_t  = d.fill_polyco_table ()

        ## add HDUs
        fits_data = fits.HDUList()
        fits_data.append (primary_hdu)
        fits_data.append (polyco_t)
        fits_data.append (table_hdu)
        fits_data.writeto(ofile, overwrite=True)




