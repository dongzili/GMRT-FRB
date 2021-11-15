
"""
classes which handle 

- only full stokes
"""
import numpy as np

import datetime as dt
from   dateutil import tz

import astropy.time  as at
import astropy.units as au
import astropy.coordinates as asc

from astropy.io import fits

__all__ = ['SNIP_SOURCES', 'MISC_SOURCES', 'RAD', 'DECD', 'DMD', 'BaseObsInfo', 'get_band', 'read_hdr', 'get_freqs', 'get_tsamp']

###################################################
SNIP_SOURCES  =  ['R3', 'R67']
MISC_SOURCES  =  ['B0329+54', '3C48', '3C147', '3C138']
SOURCES  =  SNIP_SOURCES + MISC_SOURCES

RAD      =  dict (R3=29.50312583, R67=77.01525833)
DECD     =  dict (R3=65.71675422, R67=26.06106111)
DMD      =  dict (R3=348.82, R67=411.)
RAD['3C48']      = 24.4220417
DECD['3C48']     = 33.1597417
RAD['B0329+54']  = 53.2475400
DECD['B0329+54'] = 54.5787025
DMD['B0329+54']  = 26.7641
RAD['3C147']     = 84.6812917
DECD['3C147']    = 49.8285556
RAD['3C138']     = 79.5687917
DECD['3C138']    = 16.5907806
###################################################
EDTYPE = np.float32
MODES  = ['search', 'snippet', 'cal']
MDtype = {'search':'SEARCH', 'snippet':'PSR', 'cal':'CAL'}
MDnbits= {'search':8, 'snippet':16, 'cal':8}
###################################################
def get_band (f):
    """gets beam/freq/fftint info from filename"""
    fdot   = f.split('.')
    ss     = fdot[0].split('_')
    ret    = dict (beam=ss[-5], fedge=float(ss[-4]), bw=float(ss[-3]), fftint=int(ss[-2]))
    return ret

def get_tsamp (band, nch):
    """ get tsamp in seconds """
    return band['fftint'] * nch / band['bw'] / 1E6

def get_freqs (band, nchans, lsb=False, usb=False):
    """gets faxis given band info
        The logic to generate the frequency axis is provided by Dr. R.A.Main
    """
    if lsb is True and usb is True:
        raise ValueError ("provided two sidebands")
    ##
    sbw          = abs(band['bw'])
    if lsb:
        sbw      = sbw * -1
    chan_bw      = sbw / nchans

    freqs    = np.linspace (band['fedge'], band['fedge']+sbw, nchans, endpoint=False, dtype=EDTYPE)
    freqs    += 0.5 * chan_bw
    return freqs
###################################################
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
    return at.Time (ret)
###################################################
class BaseObsInfo(object):
    """
    One class containing info which goes into 
    - search,cal-mode  psrfits
    - snippet-mode psrfits
    """

    def __init__(self, mjd, mode):
        if mode not in MODES:
            raise ValueError ("mode not understood")

        self.mode          = mode
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
        self.nbits         = MDnbits[mode]
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

        ## XXX only supporting full stokes for now
        self.npoln         = 4
        self.poln_order    = "IQUV"

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

    def fill_data_info(self, tsamp, nbins=None):
        if nbins is None and self.mode == 'snippet':
            raise ValueError ("wrong mode. snippet mode should provide nbins")
        if nbins:
            self.nbins = nbins
        else:
            self.nbins = 1
        self.nbits = MDnbits[self.mode]
        self.tsamp = tsamp 

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

    def get_lst_sub  (self, offs_sub):
        """computes LST at each of offs_sub in observation
            offs_sub in seconds
        """
        mjds  = self.start_time + offs_sub*au.second
        lst_sub = np.array ([self.__get_lst__ (m.mjd, self.longitude) for m in mjds])
        return lst_sub

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
        p_hdr["FD_HAND"] = (+1, "+/- 1. +1 is LIN:A=X,B=Y, CIRC:A=L,B=R (I)   ")

        ### XXX
        """
        polpl4 = 
        		sang=45, xyph=0, phase=1, dcc=0
        """
        p_hdr["FD_SANG"] = (45.0, "[deg] FA of E vect for equal sigma in A&B (E)  ")
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
        p_hdr["OBS_MODE"] = (MDtype[self.mode], "(PSR, CAL, SEARCH)                           ")
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
            scan_len,
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

    def fill_search_table_header(self, gulp):
        """
        Made for SEARCH mode
        """
        if self.mode == "snippet":
            raise ValueError (" reached search mode with snippet ")

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
        t_hdr["NBITS"]    = (MDnbits[self.mode], "Nr of bits/datum (SEARCH mode 'X' data, else 1)")
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

    def fill_fold_table_header(self):
        """
        Made for FOLD mode
        """
        if self.mode != "snippet":
            raise ValueError (" reached snippet mode without snippet ")

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

    def fill_polyco_table (self):
        """
        Made for FOLD mode

        its an empty header
        its a  dummy table

        fold at 1 Hz - need to check with snippet nbins and tsamp 
        fold_hz is (nbins * tsamp)**-1

        nspan is 1440 which is a day
        """
        # XXX check

        if self.mode != "snippet":
            raise ValueError (" reached snippet mode without snippet ")

        if self.nbins is None:
            raise ValueError ("nbins should have been realized")

        fold_hz = (self.nbins * self.tsamp)**-1

        t_hdr = fits.Header()
        poc_columns = [
            fits.Column(name="DATE_PRO",  format="24A", array=[[""]]),
            fits.Column(name="POLYVER",   format="16A", array=[[""]]),
            fits.Column(name="NSPAN",     format="1I",  array=[[1440]]),
            fits.Column(name="NCOEF",     format="1I",  array=[[1]]),
            fits.Column(name="NPBLK",     format="1I",  array=[[1]]),
            fits.Column(name="NSITE",     format="8A",  array=[["GMRT"]]),
            fits.Column(name="REF_FREQ",  format="1D",  unit="MHz", array=[[self.fcenter]]),

            fits.Column(name="PRED_PHS",  format="1D",  array=[[0.]]),

            fits.Column(name="REF_MJD",   format="1D",  array=[[self.start_time.mjd]]),
            fits.Column(name="REF_PHS",   format="1D",  array=[[0.]]),
            fits.Column(name="REF_F0",    format="1D",  unit="Hz", array=[[fold_hz]]),

            fits.Column(name="LGFITERR",  format="1D",  array=[[0.]]),
            fits.Column(name="COEFF",     format="15D", array=[[0.]*15]),
        ]
        # Add the columns to the table
        polyco_hdu = fits.BinTableHDU(
            fits.FITS_rec.from_columns(poc_columns), name="polyco", header=t_hdr
        )
        return polyco_hdu

