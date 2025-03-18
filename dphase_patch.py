"""
patch dphase
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits as aif


from astropy.constants import c as speed_of_light
import astropy.units as au

from shutil import copy2

_wfac = ( speed_of_light/(au.MHz) ).to(au.m).value

add_pa = lambda a, b : np.arctan ( np.tan( a + b ) )


def read_sol ( f ):
    """
    read pacv
    return freq, dphase for now
    """
    with aif.open (f) as cal:
        names = [fi.name for fi in cal]
        idx   = names.index('FEEDPAR')
        # print (f"received = {names}")
        # get SOLUTION TABLE
        cpar  = cal[idx]
    # extract
        cdata = cpar.data['DATA'].reshape ((cpar.header['NCHAN'],cpar.header['NCPAR']))
        cerr  = cpar.data['DATAERR'].reshape ((cpar.header['NCHAN'],cpar.header['NCPAR']))
        freq  = cpar.data['DAT_FREQ'][0]
        wts   = np.array (cpar.data['DAT_WTS'][0], dtype=bool)
        names = [cpar.header[f"PAR_{i:04d}"] for i in range(cpar.header['NCPAR'])]
    return wts, freq, cdata, cerr, names

def patch_sol ( f, new_dphase, efile ):
    """
    create new pacv by copying
    replace dphase with new dphase in the newly created copy

    dphase is third column
    data is (1,)
    """
    new_name = f[:-len('.pacv')] + ".patch.pacv"

    copy2 ( f, new_name )

    with aif.open ( new_name, mode='update' ) as newcal:
        names = [fi.name for fi in newcal]
        idx   = names.index('FEEDPAR')
        cpar = newcal[idx]
        #####
        cpar.data['DATA'][0,...,2] = new_dphase
        cpar.header.append (
            ('2GC', efile, "adjusting DPHASE" )
        )
    return new_name

def get_args ():
    import argparse as agp
    ap   = agp.ArgumentParser ('patch_dphase', description='Patches DPHASE of 1GC pacv solution using 1GC RM-fitted residual', epilog='Part of GMRT/FRB')
    add  = ap.add_argument
    add ('-p','--pacv', help='PACV to be patched. Will be copied', required=True, dest='pacv')
    add ('-d','--data', help='1GC Peakpkg package', required=True, dest='peakpkg')
    add ('-r','--rmfit', help='RMfit solution of the 1GC Peakpkg package', required=True, dest='rmfit')
    add ('-s','--save', help='Save patch', default=None, dest='save_patch')
    return ap.parse_args ()


if __name__ == "__main__":
    args  = get_args ()

    # peak_file  = "./B0329+54_bm1_pa_550_200_32_12nov2022.raw.calibP.peakpkg.npz"
    # rmfit_file = "./B0329+54_bm1_pa_550_200_32_12nov2022.raw.calibP.peakpkg.npz_sol.npz"
    # pacv_file  = "./FRBR3_NG_bm1_pa_550_200_32_12nov2022.raw.1.noise.ar.pacv"
    peak_file  = args.peakpkg
    rmfit_file = args.rmfit
    pacv_file  = args.pacv


    #######################################
    rmfit  = np.load ( rmfit_file )
    peak   = np.load ( peak_file )

    ####### read pacv
    pacv_wts, pacv_freqs, pacv_data, _, _ = read_sol ( pacv_file )
    pacv_dphase = pacv_data[...,2]

    ### get freqs
    ## when patching take freqs from pacv
    # freqs  = peak['fcen'] + np.linspace(-0.5*peak['fbw'], 0.5*peak['fbw'], peak['nchan'], endpoint=True)
    freqs  = pacv_freqs
    wavs   = _wfac / freqs
    wavs2  = wavs**2

    ### prepare peak
    ## Max-Min
    ## all are (npol,nchan)
    mata = np.ma.MaskedArray ( peak['max_slice'], mask=peak['mask_max'] )
    bata = np.ma.MaskedArray ( peak['min_slice'], mask=peak['mask_min'] )
    data = mata - bata
    pa_data = 0.5 * np.arctan2 ( data[2], data[1] )

    ###
    # get parameters from rmfit
    # rm, lambdac, pa, ignore lp

    fit_rm, fit_rmerr  = rmfit['rm_qu'], rmfit['rmerr_qu']
    fit_pa, fit_paerr  = np.deg2rad ( [rmfit['pa_qu'], rmfit['paerr_qu']] )
    lamc   = np.sqrt ( rmfit['l02'] )
    fit_pal  = np.arctan ( np.tan (  ( fit_rm*(wavs**2 - lamc**2) ) + fit_pa ) )
    ####################################################
    rmfit_wav2 = rmfit['lam2'] + rmfit['l02']
    rmfit_wav  = np.sqrt ( rmfit_wav2 )
    rmfit_pa   = 0.5 * np.arctan2 ( rmfit['model_u'], rmfit['model_q'] )
    rmfit_freqs = rmfit['freq_list']
    ####################################################
    epa  = add_pa ( rmfit_pa, -fit_pal )
    data_fr_res = add_pa ( pa_data, -fit_pal )
    ####################################################
    ## dphase is already inverted
    ## so add the residual to dphase
    pacv_dphase_adjust = add_pa ( pacv_dphase, -data_fr_res )
    """
    you subtract here
    here = pa_data - fit_pal
    """
    txt = f"1GC :: RM={rmfit['rm_qu']:.2f}+-{rmfit['rmerr_qu']:.2f} rad/m2 PA={rmfit['pa_qu']:.2f}+-{rmfit['paerr_qu']:.2f} deg"
    ####################################################
    if args.save_patch:
        new_name = patch_sol ( pacv_file, pacv_dphase_adjust, os.path.basename(peak_file) )
        png_name = new_name + ".png"
    ####################################################
    fig  = plt.figure ('dphase-patch')

    axd, axp = fig.subplots ( 2,1, sharex=True, sharey=True )


    # plt.plot ( wavs, fit_pal )
    # plt.plot ( rmfit_wav, rmfit_pa )
    # plt.plot ( freqs, epa )

    # axd.plot ( rmfit_freqs, rmfit_pa, marker='.',c='b', label='1GC fit' )
    axd.plot ( freqs, fit_pal, marker='.', c='b', label='1GC fit' )
    axd.plot ( freqs, pa_data, marker='.', c='r', label='1GC data' )
    axd.plot ( freqs, data_fr_res, marker='.', c='k', label='Res' )
    # plt.plot ( freqs, pa_data, marker='.', c='k' )

    # plt.plot ( wavs2, fit_pal, marker='.', c='r', ls='' )
    # plt.plot ( wavs2, pa_data, marker='.', c='k' )

    # plt.plot ( freqs, data_fr_res, marker='.', c='b', label="" )
    axp.plot ( pacv_freqs, pacv_dphase, marker='.', c='r', label='Original DPHASE' )
    axp.plot ( pacv_freqs, pacv_dphase_adjust, marker='.', c='k', label="Patched DPHASE" )

    axp.set_xlabel ('Freq / MHz')
    axp.set_ylabel ('DPHASE / rad')
    axd.set_ylabel ('DPHASE / rad')

    axp.legend(loc='best')
    axd.legend(loc='best')

    fig.suptitle ( txt )

    # plt.show ()
    if args.save_patch:
        fig.savefig ( png_name, dpi=300, bbox_inches='tight' )
    else:
        plt.show ()
