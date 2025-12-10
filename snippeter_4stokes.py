import numpy as np

import pandas as pd
import tqdm

import os
import sys

import datetime as dt
from   dateutil import tz
import astropy.time as at
import astropy.units as au

"""
Code orginally written for plotting, 
later modified for generating burst snippets
"""

NPYFILE="{tmjd:15.10f}_dm{dm:04.3f}_sn{sn:03.2f}_lof{freq:3.0f}.npz"
ndd_NPYFILE="{tmjd:15.10f}_dm{dm:04.3f}_sn{sn:03.2f}_lof{freq:3.0f}_ndd.npz"

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

########
DM       = 348.82
#DM       = 550.
DM_CONST = 4.148741601E6
def dispdelay(DM,LOFREQ,HIFREQ):
    return DM * DM_CONST * (LOFREQ**-2 - HIFREQ**-2) 

########
def old_dedisperser (IN, freq_delays):
		"""assumes de-dispersion is valid
		"""
		nsamps,nchans,nstk = IN.shape
		max_delay = freq_delays[0]
		osamps = nsamps - max_delay
		if osamps <= 0:
				raise ValueError ("incompatible de-dispersion")
		OUT = np.zeros ((osamps, nchans, nstk), dtype=IN.dtype)
		u,v = 0,0
		for ifreq, idelay in enumerate (freq_delays):
				""" low to high frequency ordering"""
				u = idelay
				v = u + osamps
				OUT[:,ifreq,:] = IN[u:v,ifreq,:]
		return OUT
## methods

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

def get_args():
		import argparse as arp
		ap = arp.ArgumentParser (prog='gmrt_4stokes_snippeter',description='Snippets',)
		add = ap.add_argument
		add ('file',help='Raw file')
		add ('--toa',help='TOA csv', required=True, dest='toa')
		add ('--dir', help='Directory to store', dest='dir', default='ia_snippets')
		add ('-v', help='Verbosity', action='store_true', dest='v')
		add ('--no-dd', help='Do not dedisperse', action='store_true', dest='ndd')
		return ap.parse_args()
		
########
def plotter (dat, SN, TIME, DIR):
		"""
			invert
		"""
		## read
		start_slice  = TIME
		start_sample = int (start_slice / tsamp)
		start_sample = start_sample - 64
		time_start   = TIME - (64 * tsamp)
		utime_start  = ato + (time_start * au.second)
		# manual override
		#width_slice  = int (WIDTH / tsamp)
		width_slice  = 64 + 64
		take_slice   = width_slice + f_delays[0]
		##
		fb   = dat[start_sample:(start_sample+take_slice),...]

		#fig.tight_layout ()
		#print (OFILE)
		if args.ndd:
				OFILE = ndd_NPYFILE.format (tmjd = utime_start.mjd, sn = SN, dm = DM, freq = freqs.min())
				np.savez (os.path.join (DIR, OFILE), fb=fb, mjd=utime_start.mjd, freqs=freqs, tsamp=tsamp)
		else:
				OFILE = NPYFILE.format (tmjd = utime_start.mjd, sn = SN, dm = DM, freq = freqs.min())
				dd   = dedisperser (fb, f_delays)
				np.savez (os.path.join (DIR, OFILE), dd=dd, mjd=utime_start.mjd, freqs=freqs, tsamp=tsamp)

########
if __name__ == "__main__":
		"""
		This is GMRT raw datafile so frequency is low to high
		"""
		args = get_args()
		##
		if not os.path.isdir (args.dir):
				os.mkdir (args.dir)
		## read raw file
		mm       = np.memmap(args.file, mode='r', dtype=np.uint16,)
		path,ext = os.path.splitext (args.file)
		HDR  = args.file + ".hdr"
		if args.v:
				print (" RAW file  = ", args.file)
				print (" HDR file  = ", HDR)

		## 
		## read hdr
		dto = read_hdr (HDR)
		ato = at.Time (dto)
		if args.v:
				print (" TSTART    = ", ato)

		tsamp    = 327.68E-6
		tsamp_ms = tsamp * 1000.
		nchans   = 2048

		BAW  = os.path.basename (args.file).split('_')
		beam = BAW[2]
		flow = BAW[3]
		fbw  = BAW[4]

		band = None
		if flow == '550':
				band = 'P'
		elif flow == '1000':
				band = 'L'
		else:
				band = 'X'
				
		if band == 'L':
				fch1         = 1200.0
				fch0         = 1000.0
		elif band == 'P':
				fch1         = 750.
				fch0         = 550.
		else:
				raise ValueError ("Only L and P are available")

		if args.v:
				print (" Beam      = ",beam)
				print (" Band      = ",band)
		#
		foff     = 0.09765625
		##
		if beam == 'ia':
				fb       = mm.reshape ((-1, nchans))
		elif beam == 'pa':
				fb       = mm.reshape ((-1, nchans, 4))
		else:
				raise IOError (f'beam not understood : {beam}')

		total_samps = fb.shape[0]
		total_time  = total_samps * tsamp

		## frequency table
		freqs =  foff * np.arange (nchans)
		freqs += fch0
		max_freq = freqs[nchans-1]

		f_delays = np.zeros (nchans, dtype=np.int64)
		for ichan, freq in enumerate (freqs):
				f_delays[ichan] = int (dispdelay (DM, freq, max_freq)/tsamp_ms)
		max_delay_s  = f_delays[0] * tsamp 

		## dm delays
		## read sps file
		toa = pd.read_csv (args.toa,)
		#it  = tqdm.tqdm (sps.index, unit='cand', desc='SPS')
		it  = tqdm.tqdm (toa.index, unit='toa', desc='Burst')

		##
		ss = os.path.basename (args.toa).split('_')
		sps_band = ss[2]
		print (band, sps_band)
		if band != sps_band:
				raise IOError (" TOA band and RAW band not matched")

		## plotting
		for i in it:
				sn = toa.sn[i]
				pt = at.Time (toa.toa[i], format='mjd') - ato
				pt_s = pt.to (au.second).value
				if pt_s < 0. or pt_s > total_time:
						raise IOError (" peak_time not in the duration")
				plotter (fb, sn, pt_s, args.dir)
		
