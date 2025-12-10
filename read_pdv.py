"""
read pdv and process and write to npz
"""
import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt

SNsfunc   = lambda x : np.polyval (
        [-1.279e-05, 4.896e-04, -7.584e-03, 6.062e-02, -2.625e-01, 5.796e-01],
        x
    )

def action ( temp, tstart, tstop, noshift=False ):
    # print ( temp, tstart, tstop )
    I, Q, U, V  = np.genfromtxt ( temp, dtype=np.float64, skip_header=True, usecols=(3,4,5,6), unpack=True )
    nbin          = I.size
    hbin          = nbin // 2

    if noshift:
        mid           = int ( 0.5 * ( tstart + tstop ) )
        shift         = hbin - mid

        # shift IQUV
        I           = np.roll ( I, shift )
        Q           = np.roll ( Q, shift )
        U           = np.roll ( U, shift )
        V           = np.roll ( V, shift )

        tstart = tstart + shift
        tstop  = tstop  + shift

    ###
    ## on phase
    ON            = np.zeros ( nbin, dtype=bool )
    OF            = np.ones ( nbin, dtype=bool )
    on_slice      = slice ( tstart, tstop )
    ON[on_slice]  = True
    ## one extract at the end
    ON[on_slice.stop+1] = True
    OF[on_slice]  = False
    pulsebins     = ON.sum()
    offbins       = OF.sum()
    ###################################
    ## ----- OFF PULSE STATISTICS -----
    I_offavg      = I[OF].mean ()
    Q_offavg      = Q[OF].mean ()
    U_offavg      = U[OF].mean ()
    V_offavg      = V[OF].mean ()

    ## removing baseline
    I_bsl         = I - I_offavg
    Q_bsl         = Q - Q_offavg
    U_bsl         = U - U_offavg
    V_bsl         = V - V_offavg

    ## computing RMS
    ## no central moment correction here?
    ## warum?
    I_rms         = np.sqrt (np.power ( I_bsl[OF], 2 ).mean ())
    # I_rms         = I_bsl[OF].std ()
    Q_rms         = np.sqrt (np.power ( Q_bsl[OF], 2 ).mean ())
    U_rms         = np.sqrt (np.power ( U_bsl[OF], 2 ).mean ())
    V_rms         = np.sqrt (np.power ( V_bsl[OF], 2 ).mean ())

    ## computing again?
    I_offavg      = I_bsl[OF].mean ()

    ###################################
    ## ----- ON PULSE STATISTICS  -----

    ## Linear polarization

    I_onavg       = I_bsl[ON].mean ()
    Q_onavg       = Q_bsl[ON].mean ()
    U_onavg       = U_bsl[ON].mean ()
    V_onavg       = V_bsl[ON].mean ()

    L             = np.sqrt ( Q**2 + U**2 )
    LIrms         = L / I_rms
    Lmask         = LIrms >= 1.57
    LON           = np.logical_and ( Lmask, ON )
    if LON.sum() <= 0:
        warnings.warn ("No Linear polarization bins", RuntimeWarning)
    Ltrue         = np.zeros_like ( L )
    ###
    Ltrue[Lmask]  = np.sqrt ( np.power ( LIrms[Lmask], 2 ) - 1.0 ) * I_rms
    L_onavg       = Ltrue[ON].mean ()

    #### PPA 
    ppa_rad             = 0.5 * np.arctan2 ( U_bsl, Q_bsl )
    ppa_deg             = np.rad2deg (ppa_rad)
    ppa_err             = np.zeros_like (ppa_deg)
    ppa_mask            = np.logical_and (LIrms >= 3, ON)
    # ppa_mask            = LON
    for i,lp0 in enumerate ( LIrms ):
        if not ppa_mask[i]:
            continue
            ## it will be taken out
            # print ('\tfocus ',end='')
        if lp0 >= 10:
            ppa_err[i] = 28.65 / lp0
            # print ('high sn')
        elif lp0 >= 3:
            ppa_err[i] = np.rad2deg (SNsfunc (lp0))
            # print ('low sn')
        else:
            raise RuntimeError (" Adjust Ltrue threshold to avoid NaN/Inf")
            ## this should not be the case

    RET = dict()
    RET['pa_deg'] = ppa_deg
    RET['paerr_deg'] = ppa_err
    RET['pa_mask'] = ppa_mask
    RET['Ltrue']   = Ltrue
    RET['I']      = I_bsl
    RET['Q']      = Q_bsl
    RET['U']      = U_bsl
    RET['V']      = V_bsl
    RET['on']     = ON

    ##
    # fig = plt.figure ('pdv')

    # plt.plot ( I_bsl )
    # plt.plot ( Q_bsl )
    # plt.plot ( U_bsl )
    # plt.plot ( V_bsl )
    # plt.plot ( Ltrue )
    # plt.plot ( ppa_deg )

    # plt.show ()

    return RET

def get_args ():
    import argparse as agp
    ag = agp.ArgumentParser ('read_pdv', description='parse pdv', epilog='Part of GMRT/FRB')
    add = ag.add_argument
    add ('pdv', help='pdv file', nargs='+')
    return ag.parse_args ()

if __name__ == "__main__":

    args = get_args ()

    for a in args.pdv:

        bn = os.path.basename (a) [:-4]

        with open (a[:-4]+".json", 'r') as f:
            ran = json.load ( f )

        tstart = ran['tstart']
        tstop  = ran['tstop']

        ret = action ( a, tstart, tstop, True )

        np.savez ( bn+".pdv.npz", **ret )

