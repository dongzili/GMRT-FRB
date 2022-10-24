"""
make_pacv

combine calibration solutions and prepare a pacv file
"""
import os
import json

import numpy as np

import datetime as dt
from   dateutil import tz

import astropy.time  as at
import astropy.units as au
import astropy.coordinates as asc

import logging

from astropy.io import fits

################################
logger = logging.getLogger (__name__)

def get_args ():
    import argparse
    agp = argparse.ArgumentParser ("make_pacv", description="Makes a pacv calibration solution file", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add ('gdg', help='Gain/diff. gain solution JSON (acquired from measure_gain_dgain.py)',)
    add ('delay', help='Delay solution JSON (acquired from measure_delay7.py)',)
    
    add ('-O', '--outdir', help='Output directory', default="./")
    add ('-d','--debug', action='store_const', const=logging.DEBUG, dest='loglevel')
    add ('-v','--verbose', action='store_const', const=logging.INFO, dest='loglevel')
    return agp.parse_args ()

class BasePacvInfo(object):
    """
    One class containing info which goes into 
    pacv calibration solution file
    """
    def __init__(self, mjd):
        self.file_date     = self.__format_date__  (at.Time.now().isot)
        self.observer      = "LGM"
        self.proj_id       = "GMRT-FRB"
        self.obs_date      = ""

        #### freq info
        self.freqs         = None
        self.fcenter       = 0.0
        self.bw            = 0.0
        self.nchan         = 0
        self.chan_bw       = 0.0

        #### source info
        self.sc            = None
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

        self.npoln         = 4
        self.poln_order    = "AABBCRCI"

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

    def fill_freq_info(self, nchans, bandwidth, freqs):
        """ uGMRT gives nchans, bandwidth and either flow or fhigh 
            All frequency units in MHz

            psrfits requires centre frequency of each channel

        """
        self.bw           = bandwidth
        self.nchan        = nchans
        self.chan_bw      = bandwidth / nchans
        self.freqs        = freqs
        self.fcenter      = self.freqs[nchans//2]

    def fill_source_info(self, src_name, rad, decd):
        """ loads src_name, RA/DEC string """
        self.sc       = asc.SkyCoord(rad, decd, unit="deg")
        ra_hms        = self.sc.ra.hms
        dec_dms       = self.sc.dec.dms
        self.src_name = src_name
        self.ra_str   = f"{int(ra_hms[0]):02d}:{np.abs(int(ra_hms[1])):02d}:{np.abs(ra_hms[2]):07.4f}"
        self.dec_str  = f"{int(dec_dms[0]):02d}:{np.abs(int(dec_dms[1])):02d}:{np.abs(dec_dms[2]):07.4f}"

    def fill_beam_info(self, beam_size):
        """ currently only support circular beams """
        self.bmaj_deg  = beam_size / 3600.0
        self.bmin_deg  = beam_size / 3600.0
        self.bpa_deg   = 0.0

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

    def fill_primary_header(self, chan_dm=0.0, scan_len=0):
        """
        Writes the primary HDU

        Need to check:
        XXX
            - FD_SANG, FD_XYPH, BE_PHASE, BE_DCC
            - beam info: BMAJ, BMIN, BPA
            - if CALIBRATION
        """
        # XXX need to check
        p_hdr = fits.Header()
        p_hdr["HDRVER"] = (
            "6.2             ",
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
        p_hdr["FD_HAND"] = (+1, "+/- 1. +1 is LIN:A=X,B=Y, CIRC:A=L,B=R (I)   ")

        ### XXX
        """
            WvS+?? psrchive+polcal paper says FD_SANG for circular feeds should be 0deg
            FD_HAND=+1 for circular feeds
        """
        p_hdr["FD_SANG"] = (0.0, "[deg] FA of E vect for equal sigma in A&B (E)  ")
        p_hdr["FD_XYPH"] = (0.0, "[deg] Phase of A^* B for injected cal (E)    ")

        p_hdr["BACKEND"]  = ("uGMRT", "Backend ID                                   ")
        p_hdr["BECONFIG"] = ("N/A", "Backend configuration file name              ")
        ### XXX
        ## BE_PHASE affects StokesV so check
        ## XXX all usb's so it should be +ive???
        p_hdr["BE_PHASE"] = (+1, "0/+1/-1 BE cross-phase:0 unknown,+/-1 std/rev")
        ## in some uGMRT bands, the top subband is taken and in some the lower subband is
        p_hdr["BE_DCC"]   = (0, "0/1 BE downconversion conjugation corrected  ")

        p_hdr["BE_DELAY"] = (0.0, "[s] Backend propn delay from digitiser input ")
        p_hdr["TCYCLE"]   = (0.0, "[s] On-line cycle time (D)                   ")

        ### PSR mode
        p_hdr["OBS_MODE"] = ("PCM", "(PSR, CAL, SEARCH)                           ")
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
        p_hdr["CHAN_DM"] = ("*", "DM used to de-disperse each channel (pc/cm^3)")

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
            scan_len,
            "[s] Requested scan length (E)                ",
        )
        ### it is FA for uGMRT
        ### CPA is super cool
        p_hdr["FD_MODE"] = ("FA", "Feed track mode - FA, CPA, SPA, TPA          ")
        p_hdr["FA_REQ"]  = (0.0, "[deg] Feed/Posn angle requested (E)          ")
        
        ### calibration 
        p_hdr["CAL_MODE"] = ("N/A", "Cal mode (OFF, SYNC, EXT1, EXT2)             ")
        p_hdr["CAL_FREQ"] = (-1.0,  "[Hz] Cal modulation frequency (E)            ")
        p_hdr["CAL_DCYC"] = (1.0,   "Cal duty cycle (E)                           ")
        p_hdr["CAL_PHS"]  = (-1.0,  "Cal phase (wrt start time) (E)               ")

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

    def fill_history_table (self):
        """
        its an empty header
        its a  dummy table
        """
        t_hdr = fits.Header()
        poc_columns = [ 
            fits.Column(name="DATE_PRO",  format="24A", array=[[self.file_date]]),
            fits.Column(name="PROC_CMD",  format="256A", array=[["UNKNOWN"]]),
            fits.Column(name="SCALE",     format="8A", array=[["FluxDen"]]),
            fits.Column(name="POL_TYPE",  format="8A", array=[["AABBCRCI"]]),

            fits.Column(name="NSUB",      format="1J",  array=[[0]]),
            fits.Column(name="NPOL",      format="1I",  array=[[4]]),
            fits.Column(name="NBIN",      format="1I",  array=[[512]]),
            fits.Column(name="NBIN_PRD",  format="1I",  array=[[512]]),
            fits.Column(name="TBIN",      format="1D",  unit="s", array=[[self.tsamp]]),
            fits.Column(name="CTR_FREQ",  format="1D",  unit="MHz", array=[[self.fcenter]]),
            fits.Column(name="NCHAN",      format="1J",  array=[[self.nchan]]),
            fits.Column(name="CHAN_BW",    format="1D",  unit="MHz", array=[[self.chan_bw]]),
            fits.Column(name="REF_FREQ",  format="1D",  unit="MHz", array=[[self.fcenter]]),

            fits.Column(name="DM",        format="1D",  unit="", array=[[0.]]),
            fits.Column(name="RM",        format="1D",  unit="", array=[[0.]]),

            fits.Column(name="PR_CORR",   format="1I",  unit="", array=[[0]]),
            fits.Column(name="FD_CORR",   format="1I",  unit="", array=[[0]]),
            fits.Column(name="BE_CORR",   format="1I",  unit="", array=[[0]]),
            fits.Column(name="RM_CORR",   format="1I",  unit="", array=[[0]]),
            fits.Column(name="DEDISP",    format="1I",  unit="", array=[[0]]),

            fits.Column(name="DDS_MTHD",    format="32A",  unit="", array=[["UNSET"]]),
            fits.Column(name="SC_MTHD",     format="32A",  unit="", array=[["NONE"]]),
            fits.Column(name="CAL_MTHD",    format="32A",  unit="", array=[["NONE"]]),
            fits.Column(name="RFI_MTHD",    format="32A",  unit="", array=[["NONE"]]),

            fits.Column(name="CAL_FILE",    format="256A",  unit="", array=[["NONE"]]),

            fits.Column(name="RM_MODEL",    format="32A",  unit="", array=[["NONE"]]),
            fits.Column(name="AUX_RM_C",   format="1I",  unit="", array=[[0]]),
            fits.Column(name="DM_MODEL",    format="32A",  unit="", array=[["NONE"]]),
            fits.Column(name="AUX_DM_C",   format="1I",  unit="", array=[[0]]),
        ]

        # Add the columns to the table
        polyco_hdu = fits.BinTableHDU(
            fits.FITS_rec.from_columns(poc_columns), name="history", header=t_hdr
        )
        return polyco_hdu

    def fill_solution_header(self,):
        """
        Put solution

        SINGLE cross coupling method
        """
        t_hdr = fits.Header()
        t_hdr['CAL_MTHD']    = ("single", "Cross coupling method")
        t_hdr['NCPAR']    = (3, "Number of coupling parameters")
        t_hdr['NCOVAR']   = (0, "Number of parameter covariances")
        t_hdr['NCHAN']    = (self.nchan, "Nr of channels in Feed coupling data")
        t_hdr['EPOCH']    = (self.start_time.mjd, "[MJD] Epoch of calibration obs")

        t_hdr["PAR_0000"]     = ("G", "scalar gain")
        t_hdr["PAR_0001"]     = ("gamma", "differential gain (hyperbolic radians)")
        t_hdr["PAR_0002"]     = ("phi", "differential phase (radians)")
        return t_hdr

if __name__ == "__main__":
    args = get_args ()
    logging.basicConfig (level=args.loglevel, format="%(asctime)s %(levelname)s %(message)s")
    #################################
    with open (args.gdg, 'r') as f:
        gdg   = json.load ( f )

    with open (args.delay, 'r') as f:
        delay = json.load ( f )
    #################################
    delay_ns  = delay['delay_ns']
    delayerr_ns = delay['delayerr_ns']
    #################################
    faxis     = gdg['freq_axis']
    nchan     = faxis['nchan']
    npar      = 3
    dat_freq  = np.array (gdg['full_freq_list'], dtype=np.float32).reshape ((1, nchan))
    dat_wts   = np.array (gdg['freq_mask'], dtype=np.float32).reshape ((1, nchan))

    data      = np.zeros ((1, nchan, npar), dtype=np.float32)
    dataerr   = np.zeros ((1, nchan, npar), dtype=np.float32)
    mask      = np.array (gdg['freq_mask'], dtype=bool)
    #################################
    data[0, mask, 0] = gdg['gain']
    data[0, mask, 1] = gdg['dgain']
    dataerr[0, mask, 0]  = gdg['gainerr']
    dataerr[0, mask, 1]  = gdg['dgainerr']
    #################################
    # print (f" Delay = {delay_ns:.6f} +- {delayerr_ns:.6f} ns")
    delay_f          = np.mod (-dat_freq * 1E-3 * delay_ns * np.pi, np.pi) - (0.5 * np.pi)
    delayerr_f       = np.mod (dat_freq * 1E-3 * delayerr_ns * np.pi, np.pi)
    kik              = np.array ([23, 33, 232, 923])
    # print (f" whylarge: {delay_f[0,kik]}, +- {delayerr_f[0,kik]}")
    data[0, mask, 2] = delay_f[0,mask]
    dataerr[0, mask, 2] = delayerr_f[0,mask]/100
    #################################
    ## using delay time measurement to keep trach of epoch
    gain_mjd  = gdg['obstime']
    delay_mjd = delay['obstime']
    gap_mjd   =  abs (gain_mjd - delay_mjd )
    if gap_mjd > 2:
        raise RuntimeError (" Delays and gains older than two days")
    dt        = at.Time ( delay_mjd, format='mjd' ).strftime ("%Y%m%d")
    outfile   = os.path.join ( args.outdir, f"mycal_{dt}.pacv" )
    pinfo     = BasePacvInfo ( delay['obstime'] )

    ##
    pinfo.fill_freq_info ( faxis['nchan'], faxis['fbw'], gdg['full_freq_list'] )
    ##
    pinfo.fill_source_info ( gdg['source'], 0., 0. )
    pinfo.fill_beam_info ( 0. )
    #################################
    primary_header  = pinfo.fill_primary_header (  )
    primary_hdu     = fits.PrimaryHDU (header=primary_header)

    history_hdu     = pinfo.fill_history_table ()

    calsol_header   = pinfo.fill_solution_header ()
    calsol_columns  = [
        fits.Column(name="DAT_FREQ", format=f"{nchan:d}D", unit="MHz", array=dat_freq),
        fits.Column(name="DAT_WTS",  format=f"{nchan:d}E", array=dat_wts),
        fits.Column(name="DATA",     format=f"{nchan*npar:d}E", array=data),
        fits.Column(name="DATAERR",  format=f"{nchan*npar:d}E", array=dataerr),
    ]
    calsol_hdu      = fits.BinTableHDU(
        fits.FITS_rec.from_columns(calsol_columns), name="feedpar", header=calsol_header
    )
    #################################
    fits_data       = fits.HDUList ()
    fits_data.append ( primary_hdu )
    fits_data.append ( history_hdu )
    fits_data.append ( calsol_hdu )
    #################################
    fits_data.writeto(outfile, overwrite=True)


