
import os
import pickle as pkl
## have to run this inside singularity ig
import psrchive

import numpy as np

def loader (
        fname,
        odir
    ):
    """
    filename
    """
    ##
    ff  = psrchive.Archive_load (fname)
    ff.convert_state ('Stokes')
    ff.remove_baseline ()
    ff.dedisperse ()
    ###
    basis = ff.get_basis()
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
    ww = np.array (ff.get_weights ().squeeze(), dtype=bool)
    wts   = np.ones (data.shape, dtype=bool)
    wts[:,:,ww,:] = False
    mata  = np.ma.array (data, mask=wts, fill_value=np.nan)
    ###
    start_time   = ff.start_time ().in_days ()
    end_time     = ff.end_time ().in_days ()
    mid_time     = 0.5 * ( start_time + end_time )
    ###
    src          = ff.get_source ()
    bname        = os.path.basename ( fname )
    ofile        = os.path.join ( odir, bname + ".np_pkl" )
    with open (ofile, 'wb') as f:
       pkl.dump (dict(
           data=data, wts=wts, fcen=fcen, fbw=fbw, nchan=nchan,
           mjd=mid_time, src=src,
           basis=basis
       ), f)

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('make_np', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('-O','--outdir', help='Output directory', dest='odir', default="./")
    add ('file', help="archive file")
    return ag.parse_args ()


if __name__ == "__main__":
    args  = get_args ()
    loader (args.file, args.odir)
