
"""
- TOA, phase
- peak flux (jy), fluence (jy-ms), isotropic energy (erg/s)
- width (ms)
- Lfraction, mean PPA angle (deg)

"""

import os
import sys
import json

import psrchive

import numpy as np

from scipy.interpolate import interp1d


SNsfunc   = lambda x : np.polyval (
        [-1.279e-05, 4.896e-04, -7.584e-03, 6.062e-02, -2.625e-01, 5.796e-01],
        x
    )

#############
def read_ar (fname, remove_baseline=True):
    """
    reads
    """
    ff    = psrchive.Archive_load (fname)
    ff.convert_state ('Stokes')
    if remove_baseline:
        ff.remove_baseline ()
    ff.dedisperse ()
    ###
    nbin  = ff.get_nbin()
    nchan = ff.get_nchan()
    dur   = ff.get_first_Integration().get_duration()
    fcen  = ff.get_centre_frequency ()
    fbw   = ff.get_bandwidth ()
    fchan = fbw / nchan
    tsamp = dur / nbin
    ###
    data  = ff.get_data ()
    #### making data and wts compatible
    ww    = np.array (ff.get_weights ().squeeze(), dtype=bool)
    wts   = np.ones (data.shape, dtype=bool)
    wts[:,:,ww,:] = False
    mata  = np.ma.array (data, mask=wts, fill_value=np.nan)
    dd    = mata[0,0]
    ### IQUV
    temp  = fname + ".temp_pdv"
    CMD   = "pdv -F -T -Z -t " + fname + " > " + temp
    os.system ( CMD )
    iquvpe  = np.genfromtxt ( temp, dtype=np.float64, skip_header=True, usecols=(3,4,5,6,7,8), unpack=True )
    os.system ( "rm -f " + temp )
    dd    = dict (nbin=nbin, nchan=nchan, dur=dur, fcen=fcen, fbw=fbw, dd=dd, iquvpe=iquvpe)
    return dd

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('plot_pkg', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-j','--json', required=False, help="JSON file containing tstart,tstop,fstart,fstop", dest='json', default=None)
    add ('file', help="archive file")
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-O','--outdir', help='Output directory', default='./', dest='odir')
    ##
    return ag.parse_args ()

if __name__ == "__main__":
    args    = get_args ()
    bn      = os.path.basename ( args.file )
    if not os.path.exists ( args.odir ):
        os.mkdir (args.odir)
    ##
    ## read file and ranges
    bf      = os.path.basename ( args.file )
    pkg     = read_ar ( args.file )

    ran = None
    if args.json:
        with open (args.json, 'rb') as f:
            ran = json.load (f)
    else:
        with open (args.file+".json", 'rb') as f:
            ran = json.load (f)
    if ran is None:
        raise RuntimeError("json not found")
    # jf      = args.file[:-7] + ".pazi.json"
    # with open (jf, 'rb') as f:

    ons     = slice ( ran['tstart'], ran['tstop'] )
    ofs     = slice ( ran['fstart'], ran['fstop'] )
    # off_s   = slice (0, 100)
    ######################################################
    dd      = pkg['dd']

    nsamp   = dd.shape[1]
    nchan   = dd.shape[0]
    tsamp   = float (pkg['dur']) / float (pkg['nbin'])

    times   = np.arange (nsamp) * tsamp
    times   -= times[nsamp//2]

    freqs   = np.linspace (-0.5*pkg['fbw'], 0.5*pkg['fbw'], nchan) + pkg['fcen']
    ######################################################
    ### IQUV profiles
    iquv    = pkg['iquvpe'][:4]
    _pa     = pkg['iquvpe'][4]

    ppa_deg = np.ma.MaskedArray ( _pa, mask=_pa==0. )
    ppa_err = np.ma.MaskedArray ( pkg['iquvpe'][5], mask=_pa==0. )

    ######################################################
    np.savez (
        os.path.join ( args.odir, bf + "_plot.npz" ),
        basename     = bf,
        times=times, freqs=freqs, ons=ons,
        iquv         = iquv,
        ppa_deg      = ppa_deg.data, 
        ppa_deg_mask = ppa_deg.mask, 
        ppa_err      = ppa_err.data, 
        ppa_err_mask = ppa_err.mask, 
        dd           = dd.data,
        dd_mask      = dd.mask
    )
