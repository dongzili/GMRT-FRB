
import os
import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# plt.style.use ('science')

import matplotlib.gridspec as mgs

from skimage.measure import block_reduce

def get_args ():
    import argparse as agp
    ag   = agp.ArgumentParser ('oneburst', epilog='Part of GMRT/FRB')
    add  = ag.add_argument
    add ('file', help="npz plot pkg")
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-O','--outdir', help='Output directory', default='./', dest='odir')
    ##
    return ag.parse_args ()

##########################################
FAC   = (16,1) # freq, time
####
def dd_process (idd):
    """preprocesses filterbank"""
    nch, nbin = idd.shape
    ### remove per channel mean/std.dev
    odd    = np.float32 (idd)
    odd    -= np.mean(odd, 1).reshape ((nch, 1))
    sshape = np.std (odd,1).reshape ((nch, 1))
    odd    = np.divide (odd, sshape, out=np.zeros_like (odd), where=sshape != 0.)
    return odd
##########################################

if __name__ == "__main__":
    args    = get_args ()
    bn      = os.path.basename ( args.file )
    if not os.path.exists ( args.odir ):
        os.mkdir (args.odir)
    ####################################################
    of       = os.path.join ( args.odir, os.path.basename ( args.file ).replace ('npz', 'png') )
    pkg      = np.load (args.file)
    ## mjd
    bf       = os.path.basename ( args.file )
    mjd      = float ( bf.split('_')[0] )
    ## unpack
    times    = pkg['times'] * 1E3
    freqs    = pkg['freqs']
    ppa_deg  = np.ma.MaskedArray ( pkg['ppa_deg'], mask=pkg['ppa_deg_mask'] )
    ppa_err  = np.ma.MaskedArray ( pkg['ppa_err'], mask=pkg['ppa_err_mask'] )
    iquv_s   = pkg['iquv']
    lub      = np.sqrt (iquv_s[1]**2 + iquv_s[2]**2)
    # lub      = np.sqrt (iquv_s[1]**2 + iquv_s[2]**2)
    nodd     = dd_process (pkg['dd'])
    dd_mask  = pkg['dd_mask']
    dd       = np.ma.MaskedArray (nodd, mask=dd_mask, fill_value=np.nan)
    ######
    odd      = block_reduce (dd, FAC, func=np.mean,)
    # odd      = dd
    ###### 
    fig      = plt.figure (dpi=300,)

    # axdd     = plt.subplot2grid ( (8,1), (4,0), rowspan=4, fig=fig,)
    # axpa     = plt.subplot2grid ( (8,1), (0,0), rowspan=2, fig=fig, sharex=axdd)
    # axpp     = plt.subplot2grid ( (8,1), (2,0), rowspan=2, fig=fig, sharex=axdd)

    axpa, axpp, axdd = fig.subplots ( 3,1, sharex=True, height_ratios=[2.0, 2.0, 4.0] )
    ##
    mean_pa    = np.mean ( ppa_deg )
    axpa.errorbar (times, ppa_deg - mean_pa, yerr=ppa_err, linestyle='', marker='d', color='b', markersize=0.4, linewidth=0.2, capsize=4)
    axpa.axhline (0., ls=':', c='k', )

    axpp.step (times, iquv_s[0], where='mid', color='k', linewidth=0.4, label='I')
    axpp.step (times, lub, where='mid', color='#ff2c00', linewidth=0.4, label='L')
    axpp.step (times, iquv_s[3], where='mid', color='#0c5da5', linewidth=0.4, label='V')

    axpp.legend (loc='upper right', fontsize=4, )

    ff = dd_mask.sum (1) != 0.0

    med      = np.median (odd)
    std      = np.std    (odd)
    vmin     = med - 1*std
    vmax     = med + 3*std
    axdd.imshow (odd, aspect='auto', cmap='plasma', extent=[times[0], times[-1], freqs[0], freqs[-1]], origin='lower', vmin=vmin, vmax=vmax, interpolation='none')

    axdd.hlines ( freqs[ff], xmin=times[0], xmax=times[-1], color='#9e9e9e', alpha=0.3 )

    # plt.axvspan (times[on_s.start], times[on_s.stop])
    axdd.set_xlabel ('Time / ms')
    axdd.set_ylabel ('Freq / MHz')
    axpp.set_ylabel ('S/N')
    axpa.set_ylabel ('PA / deg')
    # axdd.set_ylabel ('Time ')
    # plt.plot (on[0,0].mean(0))
    # axdd.get_shared_x_axes().join (axpp, axdd)
    # axdd.get_shared_x_axes().join (axpa, axdd)
    # axpp.set_xlim (-3, 4)
    # axdd.set_xlim (-5, 5)
    # axdd.set_yticks ([4.5, 5.0, 5.5])

    axpp.set_yticks ([])

    # axpp.set_xticklabels ([])
    # axpa.set_xticklabels ([])

    axpp.text (0.01, 0.95, f"{mjd:.8f}", ha='left', va='top', fontsize='x-small', transform=axpp.transAxes)
    # axpp.legend (loc='upper left')


    axdd.set_xlim (-15, 15)
    # axdd.set_xlim (-40, 40)

    # axpa.text (0.01,0.90,f"{bid.upper()}",transform=axpa.transAxes,ha='left',va='top',color='blue',size=14,bbox=dict(facecolor='white',edgecolor='none',pad=0.5))

    # axpa.text (0.95,0.95,f"{mjd:.8f}",transform=axpa.transAxes,ha='right',va='top',color='black',size=7,bbox=dict(facecolor='white',edgecolor='none',pad=0.5))

    ### xticks
    # axpp.set_xticks ([])
    # axpa.set_xticks ([])

    plt.subplots_adjust (hspace=0)

    # plt.show ()
    # plt.savefig (f"{b}.pdf", bbox_inches='tight')
    plt.savefig (of, bbox_inches='tight', dpi=700)
