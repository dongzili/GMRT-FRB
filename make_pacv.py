"""
make_pacv

write pacv


"""
import os
import json

import numpy as np

import datetime as dt
from   dateutil import tz

import astropy.time  as at
import astropy.units as au
import astropy.coordinates as asc

from astropy.io import fits

# from my_pacv import read_pkl, MyPACV
from circ_pacv import read_pkl, MyPACV

################################
RAD,DECD         = dict(),dict()
RAD['3C138']     = 79.5687917
DECD['3C138']    = 16.5907806

def get_args ():
    import argparse
    agp = argparse.ArgumentParser ("make_pacv", description="Makes a pacv calibration solution file", epilog="GMRT-FRB polarization pipeline")
    add = agp.add_argument
    add ('pkl', help='Pickle file (output of make_np.py)',)
    add ('-O', '--outdir', help='Output directory', default="./", dest='odir')
    add ('-v','--verbose', action='store_true', dest='v')
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

    def get_parallactic_angle (self, tobs):
        """ compute parallactic angle """
        import astroplan as ap
        import astropy.units as au
        observer      = ap.Observer ( location=self.el )
        pal           = observer.parallactic_angle ( tobs, self.sc ).to(au.radian)
        return pal.value

    def get_position_angle (self):
        """ compute position angle of the source
            
            Perley Butler here
            in band-4, we do not expect pos-angle to change

           -32 degrees

           This Q/U sign flip
           There is a sign flip in V
        """
        if self.src_name != "3C138":
            raise RuntimeError ("Source not identified src=",self.src_name)
        return np.deg2rad ( -32 )

    def get_rotation_measure (self):
        """ 
        Does 3C138 have RM?

        Tabara&Inoue say -2.1
        https://ui.adsabs.harvard.edu/abs/1980A%26AS...39..379T/abstract

        VLBI studies say RM is equivalent to zero
        https://ui.adsabs.harvard.edu/abs/1995A%26A...299..671D/abstract
        https://ui.adsabs.harvard.edu/abs/1997A&A...325..493C

        VLA just says take RM to be zero.
            
        """
        if self.src_name != "3C138":
            raise RuntimeError ("Source not identified src=",self.src_name)
        return -2.1

    def get_ionospheric_RM (self, tobs, duration=30., nsteps=2):
        """ uses RMextract """
        from RMextract.getRM import getRM
        ###
        pointing = [ self.sc.ra.radian, self.sc.dec.radian ]
        gmrt_pos = [self.ant_x, self.ant_y, self.ant_z]
        time     = tobs.mjd * 86400.0
        ###
        ret      = getRM ( 
            radec=pointing,
            stat_positions=[gmrt_pos],
            timestep=nsteps,
            timerange=[time,time+duration]
        )
        ###
        retrm    = ret['RM']['st1'].mean()
        return retrm

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
    ###################################################
    ### prepare files/filenames
    ###################################################
    FILE_UNCAL  = args.pkl
    base,_      = os.path.splitext ( os.path.basename ( FILE_UNCAL ) )
    undir       = os.path.join ( "mypacv_un", base )
    dpfile      = os.path.join ( args.odir, base + ".diagplot.png" )
    spfile      = os.path.join ( args.odir, base + ".solplot.png" )
    ofile       = base + ".mycal.pacv"
    outfile   = os.path.join ( args.odir, ofile  )
    ###################################################
    ### read calibrator file
    ###################################################
    ## read
    pkg, freq, sc    = read_pkl ( FILE_UNCAL )
    npar        = 3 # SingleAxis model
    nchan       = freq.size
    ## ON phase solve
    ## ON-phase is more than 60% of the maximum
    pp          = sc[0].mean(0)
    mask        = pp >= (0.60 * pp.max())
    ff          = sc[...,mask].mean(-1) - sc[...,~mask].mean(-1)
    #######################
    ## in case stokes-i (ff[0]) is negative, flag it.
    ## it should not be expected but if the calibrator scan is that bad
    ## then yea
    lz                     = ff[0] <= 0.0
    if np.any (lz):
        print (f" ON-OFF Stokes-I is negative, this should not be")
        freq.mask[lz]      = True
        sc.mask[:,lz,:]    = True
        ff.mask[...,lz]    = True
    #######################
    off_std     = sc[...,~mask].std(-1)
    #######################
    feed        = ''
    if pkg['basis'] == 'Circular':
        feed = 'CIRC'
    elif pkg['basis'] == 'Linear':
        feed = 'LIN'
    else:
        raise RuntimeError (" Basis not understood basis=",pkg['basis'])
    ###################################################
    ### prepare pacv file
    ###################################################
    tobs      = at.Time ( pkg['mjd'], format='mjd' )
    dt        = tobs.strftime ("%Y%m%d")
    mjd       = int ( pkg['mjd'] )
    pinfo     = BasePacvInfo ( pkg['mjd'] )
    pinfo.fill_freq_info ( pkg['nchan'], pkg['fbw'], freq )
    ##
    pinfo.fill_source_info ( pkg['source'], RAD[pkg['source']], DECD[pkg['source']] )
    pinfo.fill_beam_info ( 0. )
    #### parallactic angle
    pal_angle = pinfo.get_parallactic_angle ( tobs )
    #### position angle
    pos_angle = pinfo.get_position_angle ()
    #### Ionospheric RM contribution
    ionosrm     = pinfo.get_ionospheric_RM ( tobs )
    #### source RM
    srcrm       = pinfo.get_rotation_measure () 
    #### correct for both
    angle_corr  = pal_angle + pos_angle
    rm_corr     = ionosrm + srcrm
    #### logging
    print (f" Parallactic angle = {np.rad2deg(pal_angle):.3f}")
    # print (f" Position angle    = {np.rad2deg(pos_angle):.3f}")
    print (f" Ionospheric RM    = {ionosrm:.3f}")
    print (f" Source RM         = {srcrm:.3f}")
    print (f" Correction angle  = {np.rad2deg(angle_corr):.3f}")
    print (f" Correction RM     = {rm_corr:.3f}")
    ##################################################
    ### perform fitting
    ###################################################
    caler       = MyPACV ( feed, freq, ff, off_std, angle_corr, rm_corr )
    caler.fit_dphase ( undir, test=not True )
    caler.diag_plot ( dpfile )
    caler.sol_plot  ( spfile )
    ##################################################
    ### write into pacv
    ###################################################
    dat_freq  = np.array (freq, dtype=np.float32).reshape ((1, nchan))
    ## need the logical not for dat_wts
    mask      = np.logical_not ( freq.mask )
    dat_wts   = np.array (mask, dtype=np.float32).reshape ((1, nchan))
    data      = np.zeros ((1, nchan, npar), dtype=np.float32)
    dataerr   = np.zeros ((1, nchan, npar), dtype=np.float32)
    #####
    ## invert here
    ## DGAIN sign to be flipped
    ## v sensitive to AABBCRCI definition
    data[0, ..., 0] = caler.gain
    data[0, ..., 1] = caler.dgain
    data[0, ..., 2] = caler.get_line_wrap ( caler.dphase_lpar, hin=True)
    dataerr[0, ..., 0]  = caler.gainerr
    dataerr[0, ..., 1]  = caler.dgainerr
    dataerr[0, ..., 2]  = caler.dphaseerr
    ##################################################
    ### pacv fits header
    ###################################################
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


