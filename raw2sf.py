"""
Convert Filterbank files to PSRFITS file

Modified to convert uGMRT raw filterbank into PSRFITS snippets

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
import sys

import tqdm

import json
import time

import logging

import numpy as np

import datetime as dt
from   dateutil import tz

import astropy.time  as at
import astropy.units as au
import astropy.coordinates as asc

from astropy.io import fits

import matplotlib.pyplot as plt

logger = logging.getLogger (__name__)
NBITS    = 8
###################################################
SOURCES  =  ['R3', 'R67', 'B0329+54', '3C48']
RAD      =  dict (R3=29.50312583, R67=77.01525833)
DECD     =  dict (R3=65.71675422, R67=26.06106111)
DMD      =  dict (R3=348.82, R67=411.)
RAD['3C48']      = 24.4220417
DECD['3C48']     = 33.1597417
RAD['B0329+54']  = 53.2475400
DECD['B0329+54'] = 54.5787025
DMD['B0329+54']  = 26.7641
######################
def get_band (f):
    """gets band from filename"""
    fdot   = f.split('.')
    ss     = fdot[0].split('_')
    #ret    = dict (beam=ss[2], flow=float(ss[3]), bw=float(ss[4]), fftint=int(ss[5]))
    ret    = dict (beam=ss[-5], flow=float(ss[-4]), bw=float(ss[-3]), fftint=int(ss[-2]))
    return ret

class ObsInfo(object):
    """
    Class to setup observation info for psrfits header

    """

    def __init__(self, mjd, full_stokes=True, stokes_I=False):
        self.file_date     = self.__format_date__  (at.Time.now().isot)
        self.observer      = "LGM"
        self.proj_id       = "GMRT-FRB"
        self.obs_date      = ""

        #### freq info
        self.fcenter       = 0.0
        self.bw            = 0.0
        self.nchan         = 0
        self.chan_bw       = 0.0

        #### source info
        self.src_name      = ""
        self.ra_str        = "00:00:00"
        self.dec_str       = "+00:00:00"

        #### beam info
        self.bmaj_deg      = 0.0
        self.bmin_deg      = 0.0
        self.bpa_deg       = 0.0

        #### data info
        self.scan_len      = 0
        self.tsamp         = 0.0
        self.nbits         = NBITS
        self.nsuboffs      = 0.0
        self.nsblk         = 0

        ## taken from observatories.dat from tempo2
        ## 1656342.30    5797947.77      2073243.16       GMRT                gmrt
        self.telescope     = "GMRT"
        self.ant_x         = 1656342.30
        self.ant_y         = 5797947.77
        self.ant_z         = 2073243.16
        self.el            = asc.EarthLocation.from_geocentric (self.ant_x, self.ant_y, self.ant_z, unit="m")
        self.longitude     = self.el.lon.degree

        ## pol
        ### if full_stokes
        if full_stokes:
            self.npoln         = 4
            self.poln_order    = "IQUV"
        ### if stokes_I
        if stokes_I:
            self.npoln         = 1
            self.poln_order    = "AA+BB"
        if (full_stokes and stokes_I) or (not full_stokes and not stokes_I):
            raise ValueError ("Polarization not understood")

        ## mjd
        ### 
        self.start_time   = at.Time (mjd, format='mjd')
        self.stt_imjd     = int(mjd)
        stt_smjd          = (mjd - self.stt_imjd) * 24 * 3600
        self.stt_smjd     = int (stt_smjd)
        self.stt_offs     = stt_smjd - self.stt_smjd
        self.obs_date     = self.__format_date__ (self.start_time.isot)
        ### LST
        self.stt_lst      = self.__get_lst__ (mjd, self.longitude)

    def fill_freq_info(self, fcenter, nchan, chan_bw):
        self.fcenter      = fcenter
        self.bw           = nchan * chan_bw
        self.nchan        = nchan
        self.chan_bw      = chan_bw

    def fill_source_info(self, src_name, ra_str, dec_str):
        """ loads src_name, RA/DEC string """
        self.src_name   = src_name
        self.ra_str     = ra_str
        self.dec_str    = dec_str

    def fill_beam_info(self, bmaj_deg, bmin_deg, bpa_deg):
        self.bmaj_deg   = bmaj_deg
        self.bmin_deg   = bmin_deg
        self.bpa_deg    = bpa_deg

    def fill_data_info(self, tsamp, nbits):
        self.tsamp = tsamp 
        self.nbits = nbits

    def __get_lst__ (self, mjd, longitude):
        ## magic numbers
        gfac0    = 6.697374558
        gfac1    = 0.06570982441908
        gfac2    = 1.00273790935
        gfac3    = 0.000026
        mjd0     = 51544.5  # MJD at 2000 Jan 01 12h
        ##
        H        = (mjd - int(mjd)) * 24  # Hours since previous 0h
        D        = mjd - mjd0  # Days since MJD0
        D0       = int(mjd) - mjd0  # Days between MJD0 and prev 0h
        T        = D / 36525.0  # Number of centuries since MJD0
        ##
        gmst     = gfac0 + gfac1 * D0 + gfac2 * H + gfac3 * T ** 2.0
        lst      = ((gmst + longitude / 15.0) % 24.0) * 3600.0
        return lst

    def __format_date__ (self, date_str):
        # Strip out the decimal seconds
        out_str = date_str.split(".")[0]
        return out_str

    def fill_primary_header(self, chan_dm=0.0):
        """
        Writes the primary HDU

        Need to check:
            - FD_SANG, FD_XYPH, BE_PHASE, BE_DCC
            - beam info: BMAJ, BMIN, BPA
            - if CALIBRATION
        """
        p_hdr = fits.Header()
        p_hdr["HDRVER"] = (
            "5.4             ",
            "Header version                               ",
        )
        p_hdr["FITSTYPE"] = ("PSRFITS", "FITS definition for pulsar data files        ")
        p_hdr["DATE"] = (
            self.file_date,
            "File creation date (YYYY-MM-DDThh:mm:ss UTC) ",
        )
        p_hdr["OBSERVER"] = (
            self.observer,
            "Observer name(s)                             ",
        )
        p_hdr["PROJID"] = (
            self.proj_id,
            "Project name                                 ",
        )
        p_hdr["TELESCOP"] = (
            self.telescope,
            "Telescope name                               ",
        )
        p_hdr["ANT_X"] = (self.ant_x, "[m] Antenna ITRF X-coordinate (D)            ")
        p_hdr["ANT_Y"] = (self.ant_y, "[m] Antenna ITRF Y-coordinate (D)            ")
        p_hdr["ANT_Z"] = (self.ant_z, "[m] Antenna ITRF Z-coordinate (D)            ")
        p_hdr["FRONTEND"] = (
            "GWB",
            "Rx and feed ID                               ",
        )
        p_hdr["NRCVR"] = (2, "Number of receiver polarisation channels     ")
        p_hdr["FD_POLN"] = ("CIRC", "LIN or CIRC                                  ")
        p_hdr["FD_HAND"] = (-1, "+/- 1. +1 is LIN:A=X,B=Y, CIRC:A=L,B=R (I)   ")

        ### XXX
        p_hdr["FD_SANG"] = (45.0, "[deg] FA of E vect for equal sigma in A&B (E)  ")
        p_hdr["FD_XYPH"] = (0.0, "[deg] Phase of A^* B for injected cal (E)    ")

        p_hdr["BACKEND"]  = ("uGMRT", "Backend ID                                   ")
        p_hdr["BECONFIG"] = ("N/A", "Backend configuration file name              ")
        ### XXX
        ## BE_PHASE affects StokesV so check
        p_hdr["BE_PHASE"] = (-1, "0/+1/-1 BE cross-phase:0 unknown,+/-1 std/rev")
        ## in some uGMRT bands, the top subband is taken and in some the lower subband is
        p_hdr["BE_DCC"]   = (0, "0/1 BE downconversion conjugation corrected  ")

        p_hdr["BE_DELAY"] = (0.0, "[s] Backend propn delay from digitiser input ")
        p_hdr["TCYCLE"]   = (0.0, "[s] On-line cycle time (D)                   ")

        ### PSR mode
        p_hdr["OBS_MODE"] = ("PSR", "(PSR, CAL, SEARCH)                           ")
        p_hdr["DATE-OBS"] = (
            self.obs_date,
            "Date of observation (YYYY-MM-DDThh:mm:ss UTC)",
        )

        #### freq info
        p_hdr["OBSFREQ"] = (
            self.fcenter,
            "[MHz] Centre frequency for observation       ",
        )
        p_hdr["OBSBW"] = (self.bw, "[MHz] Bandwidth for observation              ")
        p_hdr["OBSNCHAN"] = (
            self.nchan,
            "Number of frequency channels (original)      ",
        )
        p_hdr["CHAN_DM"] = (chan_dm, "DM used to de-disperse each channel (pc/cm^3)")

        ### beam info
        p_hdr["BMAJ"] = (self.bmaj_deg, "[deg] Beam major axis length                 ")
        p_hdr["BMIN"] = (self.bmin_deg, "[deg] Beam minor axis length                 ")
        p_hdr["BPA"]  = (self.bpa_deg, "[deg] Beam position angle                    ")

        ## source info
        p_hdr["SRC_NAME"] = (
            self.src_name,
            "Source or scan ID                            ",
        )
        p_hdr["COORD_MD"] = ("J2000", "Coordinate mode (J2000, GAL, ECLIP, etc.)    ")
        p_hdr["EQUINOX"]  = (2000.0, "Equinox of coords (e.g. 2000.0)              ")
        p_hdr["RA"]       = (self.ra_str, "Right ascension (hh:mm:ss.ssss)              ")
        p_hdr["DEC"]      = (self.dec_str, "Declination (-dd:mm:ss.sss)                  ")
        p_hdr["STT_CRD1"] = (
            self.ra_str,
            "Start coord 1 (hh:mm:ss.sss or ddd.ddd)      ",
        )
        p_hdr["STT_CRD2"] = (
            self.dec_str,
            "Start coord 2 (-dd:mm:ss.sss or -dd.ddd)     ",
        )
        p_hdr["TRK_MODE"] = ("TRACK", "Track mode (TRACK, SCANGC, SCANLAT)          ")
        p_hdr["STP_CRD1"] = (
            self.ra_str,
            "Stop coord 1 (hh:mm:ss.sss or ddd.ddd)       ",
        )
        p_hdr["STP_CRD2"] = (
            self.dec_str,
            "Stop coord 2 (-dd:mm:ss.sss or -dd.ddd)      ",
        )
        p_hdr["SCANLEN"] = (
            self.scan_len,
            "[s] Requested scan length (E)                ",
        )
        ### it is FA for uGMRT
        ### CPA is super cool
        p_hdr["FD_MODE"] = ("FA", "Feed track mode - FA, CPA, SPA, TPA          ")
        p_hdr["FA_REQ"]  = (0.0, "[deg] Feed/Posn angle requested (E)          ")
        
        ### calibration 
        p_hdr["CAL_MODE"] = ("OFF", "Cal mode (OFF, SYNC, EXT1, EXT2)             ")
        p_hdr["CAL_FREQ"] = (0.0, "[Hz] Cal modulation frequency (E)            ")
        p_hdr["CAL_DCYC"] = (0.0, "Cal duty cycle (E)                           ")
        p_hdr["CAL_PHS"]  = (0.0, "Cal phase (wrt start time) (E)               ")

        ### dates
        p_hdr["STT_IMJD"] = (
            self.stt_imjd,
            "Start MJD (UTC days) (J - long integer)      ",
        )
        p_hdr["STT_SMJD"] = (
            self.stt_smjd,
            "[s] Start time (sec past UTC 00h) (J)        ",
        )
        p_hdr["STT_OFFS"] = (
            self.stt_offs,
            "[s] Start time offset (D)                    ",
        )
        p_hdr["STT_LST"] = (
            self.stt_lst,
            "[s] Start LST (D)                            ",
        )
        return p_hdr

    def fill_table_header(self, gulp):
        """
        Made for SEARCH mode
        """
        t_hdr = fits.Header()
        t_hdr["INT_TYPE"] = ("TIME", "Time axis (TIME, BINPHSPERI, BINLNGASC, etc)   ")
        t_hdr["INT_UNIT"] = ("SEC", "Unit of time axis (SEC, PHS (0-1), DEG)        ")
        t_hdr["SCALE"]    = ("FluxDen", "Intensity units (FluxDen/RefFlux/Jansky)       ")
        t_hdr["NPOL"]     = (self.npoln, "Nr of polarisations                            ")
        t_hdr["POL_TYPE"] = (
            self.poln_order,
            "Polarisation identifier (e.g., AABBCRCI, AA+BB)",
        )
        t_hdr["TBIN"]     = (self.tsamp, "[s] Time per bin or sample                     ")
        t_hdr["NBIN"]     = (1, "Nr of bins (PSR/CAL mode; else 1)              ")
        t_hdr["NBIN_PRD"] = (0, "Nr of bins/pulse period (for gated data)       ")
        t_hdr["PHS_OFFS"] = (0.0, "Phase offset of bin 0 for gated data           ")
        t_hdr["NBITS"]    = (NBITS, "Nr of bits/datum (SEARCH mode 'X' data, else 1)")
        t_hdr["NSUBOFFS"] = (
            0,
            "Subint offset (Contiguous SEARCH-mode files)   ",
        )
        t_hdr["NCHAN"]    = (self.nchan, "Number of channels/sub-bands in this file      ")
        t_hdr["CHAN_BW"]  = (
            self.chan_bw,
            "[MHz] Channel/sub-band width                   ",
        )
        t_hdr["NCHNOFFS"] = (0, "Channel/sub-band offset for split files        ")
        t_hdr["NSBLK"]    = (gulp, "Samples/row (SEARCH mode, else 1)              ")
        return t_hdr

"""
fold mode data in 16bit signed after scale, offset removal

XXX - dat shape is (npol, nchans, nbins) because FITS like to change the order
## unsigned to signed
work_data         = np.float64 (dd)
work_data[work_data > 2**15] -= 2**16

## XXX add RFI excision logic here and work on `work_data`
dat_wts[:, :50]   = 0.
dat_wts[:, -50:]  = 0.

## stokes IQUV
stoke_data        = np.zeros_like (work_data)
stoke_data[...,0] = 0.5 * (work_data[...,0] + work_data[...,2])
stoke_data[...,3] = 0.5 * (work_data[...,0] - work_data[...,2])
stoke_data[...,1] = work_data[...,1]
stoke_data[...,2] = work_data[...,2]

if False:
## data is already 16I so i think scaling/offsetting is not needed
off               = stoke_data.mean(0)
dat_offs[0, ...]  = off
stoke_data       -= off

scl_min           = stoke_data.min (0)
scl_max           = stoke_data.max (0)
# stoke_data       /= 


dat[0]             = np.int16 (stoke_data.T)
"""

TIME_FMT = "IST Time: %H:%M:%S.%f"
DATE_FMT = "Date: %d:%m:%Y"

def read_hdr(f):
    """ Reads HDR file and returns a datetime object"""
    with open (f, 'r') as ff:
        hdr = [a.strip() for a in ff.readlines()]
        hdr = filter(lambda x : not x.startswith ('#'), hdr)
    ist, date = None,None
    for h in hdr:
        if h.startswith ("IST"):
            ist = dt.datetime.strptime (h[:-3], TIME_FMT)
                # slicing at the end because we only get microsecond precision
        elif h.startswith ("Date"):
            date = dt.datetime.strptime (h, DATE_FMT)
    ret = dt.datetime (date.year, date.month, date.day, ist.hour, ist.minute, ist.second, ist.microsecond, tzinfo=tz.gettz('Asia/Calcutta'))
    return ret

def get_args ():
    import argparse
    agp = argparse.ArgumentParser ("raw2sf", description="Converts uGMRT raw to search-mode PSRFITS", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add ('-c,--nchan', help='Number of channels', type=int, required=True, dest='nchans')
    add ('-p,--npol', help='Number of polarization', type=int, required=True, dest='npol')
    add ('-b,--bit-shift', help='Bitshift', type=int, dest='bitshift', default=6)
    add ('--lsb', help='Lower subband', action='store_true', dest='lsb')
    add ('--usb', help='Upper subband', action='store_true', dest='usb')
    add ('--gulp', help='Samples in a block', dest='gulp', default=2048, type=int)
    add ('--beam-size', help='Beam size in arcsec', dest='beam_size', default=4, type=float)
    add ('-s', '--source', help='Source', choices=SOURCES, required=True)
    add ('-O', '--outdir', help='Output directory', default="./")
    add ('-d','--debug', action='store_const', const=logging.DEBUG, dest='loglevel')
    add ('-v','--verbose', action='store_const', const=logging.INFO, dest='loglevel')
    add ('raw', help='Path to raw file',)
    return agp.parse_args ()

if __name__ == "__main__":
    args = get_args ()
    logging.basicConfig (level=args.loglevel, format="%(asctime)s %(levelname)s %(message)s")
    #################################
    if args.npol != 4:
        raise ValueError (" currently only support full stokes ")
    #################################
    # subband logic
    if args.lsb and args.usb:
        raise ValueError (" cannot specify both subbands ")
    if not (args.lsb or args.usb):
        raise ValueError (" must specify atleast one subband ")
    #################################
    GULP = args.gulp
    nch  = args.nchans
    npl  = args.npol
    raw  = args.raw
    baw  = os.path.basename (raw)
    hdr  = raw + ".hdr"
    logging.info (f"Raw file           = {raw}")
    logging.info (f"Raw header file    = {hdr}")
    ### read time
    rawt = at.Time (read_hdr (hdr))
    logging.info (f"Raw MJD            = {rawt.mjd:.5f}")
    ### read raw
    rfb  = np.memmap (raw, dtype=np.uint16, mode='r', offset=0, )
    fb   = rfb.reshape ((-1, nch, npl))
    logging.debug (f"Raw shape         = {fb.shape}")
    ### read freq/tsamp
    band = get_band (baw)
    foff = band['bw'] / nch
    tsamp= band['fftint'] * nch / band['bw'] / 1E6
    freqs= None
    if args.usb:
        freqs = np.linspace (band['flow'], band['flow']+band['bw'], nch)
    if args.lsb:
        freqs = np.linspace (band['flow']-band['bw'], band['flow'], nch)
    logging.debug (f"Frequencies       = {freqs[0]:.0f} ... {freqs[-1]:.0f}")
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

    ## freq
    fch1    = freqs[0]
    foff    = freqs[1] - freqs[0]
    fcenter = freqs[nch//2]

    # Source Info
    sc      = asc.SkyCoord(RAD[args.source], DECD[args.source], unit="deg")
    ra_hms  = sc.ra.hms
    dec_dms = sc.dec.dms
    ra_str = (
        f"{int(ra_hms[0]):02d}:{np.abs(int(ra_hms[1])):02d}:{np.abs(ra_hms[2]):07.4f}"
    )
    dec_str = f"{int(dec_dms[0]):02d}:{np.abs(int(dec_dms[1])):02d}:{np.abs(dec_dms[2]):07.4f}"

    # Beam Info
    bmaj_deg  = args.beam_size / 3600.0
    bmin_deg  = args.beam_size / 3600.0
    bpa_deg   = 0.0

    # Fill in the ObsInfo class
    d = ObsInfo (rawt.mjd, npl)
    d.fill_freq_info (fcenter, nch, foff)
    d.fill_source_info (args.source, ra_str, dec_str)
    d.fill_beam_info (bmaj_deg, bmin_deg, bpa_deg)
    d.fill_data_info (tsamp, NBITS) # read head

    # subint columns
    t_row     = GULP * tsamp
    ## XXX not considering the half row
    d.scan_len= t_row * nrows 
    ## XXX not considering the half row

    tsubint   = np.ones (nrows, dtype=np.float64) * t_row
    offs_sub  = (np.arange(nrows) + 0.5) * t_row

    lst_sub   = np.array( [d.__get_lst__(rawt.mjd + tsub / (24.0 * 3600.0), d.longitude) for tsub in offs_sub],dtype=np.float64,)

    ra_deg, dec_deg   = sc.ra.degree, sc.dec.degree
    scg               = sc.galactic
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
    phdr      = d.fill_primary_header(chan_dm=0.)
    subinthdr = d.fill_table_header(GULP)
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
    udat     = np.zeros ((GULP, nch, npl), dtype=np.int16)
    pdat     = np.zeros ((nch, npl, GULP), dtype=np.uint16)
    for i in tr:
        udat[:] = 0
        pdat[:] = 0
        pkg    = fb[i:(i+GULP)]
        ### data wrangling
        """
        data ordering : (nchans, npol, nsblk*nbits/8)

        DATA is already in nbit=16
        
        pkg shape = (nsamps, nchans, npol)
        dat shape = (nchans, npol  , nsamps)
        """
        ###
        ## GMRT pol order to full stokes IQUV
        ## may need to complicate to support total intensity
        udat[...,0] = pkg[...,0] + pkg[...,2]
        udat[...,1] = pkg[...,1]
        udat[...,2] = pkg[...,3]
        udat[...,3] = pkg[...,0] - pkg[...,2]

        ###
        ## axis ordering
        pdat[:] = np.moveaxis (udat, 0, -1)

        subint_sf.data[isubint]['DATA'] = np.uint8 (pdat.T[:] >> args.bitshift)
        isubint = isubint + 1

        ## flush?
        write_sf.flush ()

    ##
    write_sf.close ()

