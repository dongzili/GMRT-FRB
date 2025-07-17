"""
pacv solution
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as aif

import scipy.optimize as so

##############################
SOL = 'pacv_sols/3C48_NGON_bm3_pa_550_200_32_16mar2021.raw.noise.ar.pazi.pacv'
def get_args ():
    import argparse as agp
    ap   = agp.ArgumentParser ('fit_delay', description='Fits delay from pacv solution', epilog='Part of GMRT/FRB')
    add  = ap.add_argument
    add ('pacv', help='PACV solution FITS file', nargs='+')
    add ('-o','--opng', help='Save PNG file', default=None, dest='save_png')
    return ap.parse_args ()

def read_sol (f):
    """reads pacv and returns frequencies and parameters and their names"""
    cal   = aif.open (f)
    names = [fi.name for fi in cal]
    # print (f"received = {names}")
    # get SOLUTION TABLE
    cpar  = cal[2]
    # extract
    cdata = cpar.data['DATA'].reshape ((cpar.header['NCHAN'],cpar.header['NCPAR']))
    cerr  = cpar.data['DATAERR'].reshape ((cpar.header['NCHAN'],cpar.header['NCPAR']))
    freq  = cpar.data['DAT_FREQ'][0]
    wts   = np.array (cpar.data['DAT_WTS'][0], dtype=bool)
    names = [cpar.header[f"PAR_{i:04d}"] for i in range(cpar.header['NCPAR'])]
    return freq[wts], cdata[wts], cerr[wts], names

# def fit (x, delay, psi):
    # """ fits pi*delay*freq + psi to phi parameter 
        
        # delay provided should be in log10
        # psi is in radians and should be in [-0.5 pi, 0.5 pi]
    # """
    # return psi - (np.power(10.0,delay) * np.pi * x)
##############################

if __name__ == "__main__":

    args          = get_args ()

    fig     = plt.figure ()
    axl,axe = fig.subplots (2, 1, sharex=True)

    for pacv,COL in zip ( args.pacv, list("rgbm") ):
        fw, sw, swerr, names = read_sol ( pacv )
        abs_gain      = sw[...,0]
        diff_gain     = sw[...,1]
        abs_gain_err  = swerr[...,0]
        diff_gain_err = swerr[...,1]
        ## got to unwrap the diff_phase
        diff_phase    = np.unwrap (sw[...,2], period=np.pi)
        diff_phase_err= np.unwrap (swerr[...,2], period=np.pi)
        # print (f" received parameters = {names}")
        # print (f" received nchans = {fw.size}")
        fw_mhz        = fw * 1E6 

        # print (f" calling curve_fit ... ", end='')
        # popt, pcov    = so.curve_fit (fit, fw_mhz, diff_phase, p0=[-7.5, 0.25*np.pi],bounds=([-8, -0.5*np.pi],[-6, 0.5*np.pi]), sigma=diff_phase_err, absolute_sigma=False)
        popt, pcov    = np.polyfit ( fw_mhz, diff_phase, 1, cov=True )
        # print (popt)
        perr          = np.sqrt ( np.diag (pcov) )
        # print (f" done")
        ##############################
        delay_ns      = popt[0] * 1E9 / np.pi
        delay_ns_err  = perr[0] * 1E9 / np.pi
        bias          = popt[1]
        bias_err      = perr[1]
        STR           = f"Delay = {delay_ns:.3f} +- {delay_ns_err:.3f} ns\nbias = {bias:.3f} +- {bias_err:.3f}"
        # print (STR)
        ##############################
        # fittedw       = fit ( fw_mhz, *popt )
        fittedw       = popt[1] + (popt[0] * fw_mhz)
        ##############################


        axl.scatter (fw, diff_phase, s=3, marker='s', color=COL, label=os.path.basename(pacv), edgecolor='k')

        axl.plot (fw, fittedw, markersize=3, color=COL, label=STR)

        axe.plot ( fw, diff_phase - fittedw , color=COL )

    axe.set_xlabel ('Frequency / MHz')
    axl.set_ylabel ('Phi unwraped / radians')
    axe.set_ylabel ('Error phi / radians')

    axl.legend (loc='best')

    # fig.suptitle ( STR )

    if args.save_png:
        fig.savefig (args.save_png, dpi=300, bbox_inches='tight')
    else:
        plt.show ()

