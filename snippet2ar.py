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

import json
import time

import logging

import numpy as np

import astropy.time  as at
import astropy.units as au
import astropy.coordinates as asc

from astropy.io import fits

import matplotlib.pyplot as plt

logger = logging.getLogger (__name__)
###################################################
TSAMP    = 327.68E-6
NCHANS   = 2048
#####################
SOURCES  =  ['R3', 'R67']
RAD      =  dict (R3=29.50312583, R67=77.01525833)
DECD     =  dict (R3=65.71675422, R67=26.06106111)
###################################################
def band2freq (bandid, nchans=NCHANS):
    """band number to frequency"""
    if bandid == 5:
        return np.linspace (1000., 1200., nchans, dtype=np.float32)
    elif bandid == 4:
        return np.linspace (550., 750., nchans, dtype=np.float32)
    elif bandid == 3:
        return np.linspace (500., 300., nchans, dtype=np.float32)
    else:
        raise ValueError (f"Band id not understood input={bandid}")

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
        self.nbits         = 16
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

    def fill_data_info(self, tsamp, nbins, nbits):
        self.tsamp = tsamp 
        self.nbins = nbins
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

    def fill_table_header(self):
        """
        Made for FOLD mode
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
        t_hdr["NBIN"]     = (self.nbins, "Nr of bins (PSR/CAL mode; else 1)              ")
        t_hdr["NBIN_PRD"] = (0, "Nr of bins/pulse period (for gated data)       ")
        t_hdr["PHS_OFFS"] = (0.0, "Phase offset of bin 0 for gated data           ")
        t_hdr["NBITS"]    = (1, "Nr of bits/datum (SEARCH mode 'X' data, else 1)")
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
        t_hdr["NSBLK"]    = (1, "Samples/row (SEARCH mode, else 1)              ")
        return t_hdr

    def fill_polyco_header(self):
        """
        Made for FOLD mode

        its an empty header

        nspan is 1440 which is a day
        """
        t_hdr = fits.Header()
        return t_hdr

def dtype_label (dtype):
    """ numpy dtype to character label """
    data_type = None
    if dtype == np.uint8:
        data_format = "B"
    elif dtype == np.int16:
        data_format = "I"
    elif dtype == np.int32:
        data_format = "J"
    elif dtype == np.int64:
        data_format = "K"
    elif dtype == np.float32:
        data_format = "E"
    elif dtype == np.float64:
        data_format = "D"
    else:
        data_format = "E"
    return data_format

def write_psrfits(
    outfile,
    dd,
    mjd,
    tsamp,
    freqs,
    source_name,
    ra, dec,
    beam_size=4, # arcsec
):
    """
    
    Args:
        outfile: str
            Path to the output PSRFITS file
        dd: numpy.array (nbins, nchans, npol)
            De-dispersed filterbank
        mjd: float
            MJD timestamp of the first sample of the dd
        tsamp: float
            Sampling time in [s]
        freqs: numpy.array (nchans,)
            Frequency axis


    Set up a PSRFITS file with everything set up EXCEPT
    the DATA.

    Args:
        outfile: path to the output fits file to write to
        your_object: your object with the input Filterbank file
        npsub: number of spectra in a subint
        nstart: start sample to read from (for the input file)
        nsamp: number of spectra to read
        chan_freqs: array with frequencies of all the channels
        npoln: number of polarisations in the output file
        poln_order: polsarisation order

    """

    nbins      = dd.shape[0]
    nchans     = dd.shape[1]
    if len(dd.shape) == 3:
        npol   = dd.shape[2]
    else:
        npol   = 1

    if nchans != freqs.size:
        raise ValueError (" data nchans and frequency axis not equal")

    fch1    = freqs[0]
    foff    = freqs[1] - freqs[0]
    fcenter = freqs[nchans//2]

    # Source Info
    sc      = asc.SkyCoord(ra, dec, unit="deg")
    ra_hms  = sc.ra.hms
    dec_dms = sc.dec.dms
    ra_str = (
        f"{int(ra_hms[0]):02d}:{np.abs(int(ra_hms[1])):02d}:{np.abs(ra_hms[2]):07.4f}"
    )
    dec_str = f"{int(dec_dms[0]):02d}:{np.abs(int(dec_dms[1])):02d}:{np.abs(dec_dms[2]):07.4f}"

    # Beam Info
    bmaj_deg  = beam_size    / 3600.0
    bmin_deg  = beam_size    / 3600.0
    bpa_deg   = 0.0

    # Fill in the ObsInfo class
    d = ObsInfo (mjd, full_stokes=(npol==4), stokes_I=(npol==1))
    d.fill_freq_info (fcenter, nchans, foff)
    d.fill_source_info (source_name, ra_str, dec_str)
    d.fill_beam_info (bmaj_deg, bmin_deg, bpa_deg)
    d.fill_data_info (tsamp, nbins, 16) # read head

    logging.info("ObsInfo updated with relevant parameters")

    ## fold mode, only one burst ==> only one subint
    n_subints = 1

    tstart = 0.0
    t_subint = nbins * tsamp
    d.scan_len = t_subint * n_subints

    tsubint   = np.ones(n_subints, dtype=np.float64) * t_subint
    offs_sub  = (np.arange(n_subints) + 0.5) * t_subint + tstart

    # logger.info(
        # f"Setting the following info to be written in {outfile} \n {json.dumps(vars(d), indent=4, sort_keys=True)}"
    # )

    # Fill in the headers
    phdr      = d.fill_primary_header()
    thdr      = d.fill_table_header()
    ohdr      = d.fill_polyco_header()
    fits_data = fits.HDUList()

    logging.info("Building the POLYCO table")
    nrow      = 1
    poc_columns = [
        fits.Column(name="DATE_PRO",  format="24A", array=[[""]]),
        fits.Column(name="POLYVER",   format="16A", array=[[""]]),
        fits.Column(name="NSPAN",     format="1I",  array=[[1440]]),
        fits.Column(name="NCOEF",     format="1I",  array=[[1]]),
        fits.Column(name="NPBLK",     format="1I",  array=[[1]]),
        fits.Column(name="NSITE",     format="8A",  array=[["GMRT"]]),
        fits.Column(name="REF_FREQ",  format="1D",  unit="MHz", array=[[fcenter]]),

        fits.Column(name="PRED_PHS",  format="1D",  array=[[0.]]),

        fits.Column(name="REF_MJD",   format="1D",  array=[[mjd]]),
        fits.Column(name="REF_PHS",   format="1D",  array=[[0.]]),
        fits.Column(name="REF_F0",    format="1D",  unit="Hz", array=[[1.]]),

        fits.Column(name="LGFITERR",  format="1D",  array=[[0.]]),
        fits.Column(name="COEFF",     format="15D", array=[[0.]*15]),
    ]
    # Add the columns to the table
    polyco_hdu = fits.BinTableHDU(
        fits.FITS_rec.from_columns(poc_columns), name="polyco", header=ohdr
    )

    logging.info("Building the PSRFITS table")
    # Prepare arrays for columns
    lst_sub   = np.array(
        [d.__get_lst__(mjd + tsub / (24.0 * 3600.0), d.longitude) for tsub in offs_sub],
        dtype=np.float64,
    )
    ra_deg, dec_deg   = sc.ra.degree, sc.dec.degree
    scg               = sc.galactic
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

    dat_wts           = np.ones((n_subints, nchans), dtype=np.float32)
    dat_offs          = np.zeros((n_subints, nchans, npol), dtype=np.float32)
    dat_scl           = np.ones((n_subints, nchans, npol), dtype=np.float32)
    # dat               = np.zeros ((n_subints, nbins, nchans, npol), dtype=np.int16)
    dat               = np.zeros ((n_subints, npol, nchans, nbins), dtype=np.int16)

    """
    fold mode data in 16bit signed after scale, offset removal

    XXX - dat shape is (npol, nchans, nbins) because FITS like to change the order
    """
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

    # fig = plt.figure ()
    # plt.plot (dat[0,...,0].mean(1))
    # plt.show ()

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
    table_hdu = fits.BinTableHDU(
        fits.FITS_rec.from_columns(tbl_columns), name="subint", header=thdr
    )

    # Add primary header
    primary_hdu = fits.PrimaryHDU(header=phdr)

    # Add hdus to FITS file and write
    logging.info(f"Writing PSRFITS table to file: {outfile}")
    fits_data.append(primary_hdu)
    fits_data.append (polyco_hdu)
    fits_data.append(table_hdu)
    fits_data.writeto(outfile, overwrite=True)
    logging.info(f"Header information written in {outfile}")
    return

def get_args ():
    import argparse
    agp = argparse.ArgumentParser ("2ar", description="Converts burst snippets to PSRFITS", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add ('-v', '--verbose', action='store_true', help='Verbose switch')
    add ('-b', '--band', help='Band', type=int, required=True)
    add ('-s', '--source', help='Source', choices=SOURCES, required=True)
    add ('-O', '--outdir', help='Output directory', default="./")
    add ('snippets', nargs='*', help='Burst snippets npz-files',)
    return agp.parse_args ()

if __name__ == "__main__":
    args = get_args ()
    freq = band2freq (args.band)
    aa   = [TSAMP, freq, args.source, RAD[args.source], DECD[args.source]]
    #################################
    for s in args.snippets:
        bs   = os.path.basename (s)
        out  = os.path.join (args.outdir, bs.replace('npz', 'ar'))
        if args.verbose:
            print (f"\nInput={bs}\tOutput={out}")
        ############################
        ff   = np.load (s)
        write_psrfits (out, ff['dd'], ff['mjd'], 
            *aa
        )
