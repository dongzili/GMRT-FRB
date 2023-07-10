"""
calculate PF

Heavily based on Aris Noustsos's C++ code

Uses psrchive pdv
so will be running inside dspsr

PA measurements use the Everett+Weinstein thingy

output a json file
"""
import os
import json

import warnings

import numpy as np

# from scipy.interpolate import interp1d
# PAerr_pkg = np.load ("lowSN_PAerror.npz")
"""
Instead of using that integration to get errors
i am fitting a polynomial to it and using the polynomial
should not make any difference tbh

SNsfunc   = interp1d (
    np.linspace (3., 10., 1024),
    np.polyval (
        [-1.279e-05, 4.896e-04, -7.584e-03, 6.062e-02, -2.625e-01, 5.796e-01],
        np.linspace ( 3., 10., 1024 )
    )
)
"""
SNsfunc   = lambda x : np.polyval (
        [-1.279e-05, 4.896e-04, -7.584e-03, 6.062e-02, -2.625e-01, 5.796e-01],
        x
    )
def jl (l):
    return [float(il) for il in l]

def weighted_mean ( m, e ):
    """ weighted mean where weights are errors """
    w  = np.power ( e, -2.0 )
    me  = np.sum ( m * w ) / np.sum ( w )
    ee  = np.sqrt ( 1.0 / np.sum (w) )
    return me, ee

def read_ar (fname):
    """
    reads
    """
    ff    = psrchive.Archive_load (fname)
    ff.convert_state ('Stokes')
    ff.remove_baseline ()
    ff.dedisperse ()
    ###
    nbin  = ff.get_nbin()
    nchan = ff.get_nchan()
    # dur   = ff.get_first_Integration().get_duration()
    dur   = ff.get_first_Integration().get_folding_period ()
    fcen  = ff.get_centre_frequency ()
    fbw   = ff.get_bandwidth ()
    fchan = fbw / nchan
    tsamp = dur / nbin
    ###
    start_time   = ff.start_time ().in_days ()
    end_time     = ff.end_time ().in_days ()
    mid_time     = 0.5 * ( start_time + end_time )
    ###
    src          = ff.get_source ()
    rmc          = ff.get_faraday_corrected ()
    pcal         = ff.get_poln_calibrated ()
    ###
    data  = ff.get_data ()
    #### making data and wts compatible
    ww = np.array (ff.get_weights ().squeeze(), dtype=bool)
    wts   = np.ones (data.shape, dtype=bool)
    wts[:,:,ww,:] = False
    mata  = np.ma.array (data, mask=wts, fill_value=np.nan)
    dd    = dict (nbin=nbin, nchan=nchan, dur=dur, fcen=fcen, fbw=fbw, 
        data=data, 
        # wts = wts, 
        wts = ww,
        rm_correction = rmc,
        polcal = pcal,
        # wts = jl ( ww ),
        # data_shape = jl ( data.shape ),
        # data_ravel = jl ( data.ravel () ),
        tsamp=tsamp, src=src, obstime=mid_time)
    return dd


def get_args ():
    import argparse as agp
    arg   =  agp.ArgumentParser ('measure_pol', description='To measure Linear and Circular polarization and Position Angle', epilog='Part of GMRT/FRB')
    add   =  arg.add_argument
    add ('ar', help='Calibrated RM corrected burst archive',)
    add ('-v','--verbose', help='Verbose', action='store_true', dest='v')
    add ('-j','--json', default=None, help="JSON file containing tstart,tstop,fstart,fstop", dest='json')
    add ('-O','--outdir', help='Output directory', dest='odir', default="./")
    return arg.parse_args ()

if __name__ == "__main__":
    RET     = dict()
    args    = get_args ()
    ####
    if not os.path.exists ( args.odir ):
        os.mkdir ( args.odir )
    ####
    pnf,_   = os.path.splitext ( args.ar )
    bn      = os.path.basename ( args.ar )
    dn      = os.path.dirname ( args.ar )
    bnf,_   = os.path.splitext ( bn )
    onf     = os.path.join ( args.odir, bnf )
    if args.odir:
        outfile = os.path.join ( args.odir, bnf + ".pol_json" )
    else:
        outfile =  args.file + ".pol_json"
    #############################################
    ## read JSON
    if args.json:
        with open (args.json, 'rb') as f:
            ran = json.load (f)
        RET['json'] = args.json
    else:
        ## this Cz.json because that naming convention
        with open (pnf+".Cz.json", 'r') as f:
            ran = json.load (f)
        RET['json'] = pnf + ".Cz.json"
    ##
    #############################################
    ## prepare to read
    semp    = onf + ".tmp"
    remp    = onf + ".rm"
    temp    = onf + "_tmp.iquv"
    RET['mpol_ar']  = bn
    ###
    if os.path.exists ( temp ):
        os.system ("rm -rf " + temp)
    #############
    ### PSRCHIVE calls here
    """
    to convert to Stokes
    pam -S -u ./ -e tmp {input}

    to get RM
    psrstat -c rm,rmc {semp}

    to make IQUV time series
    pdv -F -T -t {semp} >> {temp}
    """
    cmd_to_stokes   = "pam -S -u " + args.odir + " -e tmp " + args.ar
    cmd_to_get_rm   = "psrstat -c rm,rmc -Q " + semp +  " > " + remp
    cmd_to_get_data = "pdv -F -T -t " + semp + " >> " + temp
    cmd_to_clear    = "rm -f " + semp + " " + temp + " " + remp
    ###### delete after reading
    os.system ( cmd_to_stokes )
    os.system ( cmd_to_get_rm )
    _,rm,rmc        = np.genfromtxt ( remp, dtype=np.float64, unpack=True )
    if rmc != 1:
        raise RuntimeError (" The file is not RM corrected!!")
    RET['rm']       = rm
    os.system ( cmd_to_get_data )
    I, Q, U, V  = np.genfromtxt ( temp, dtype=np.float64, skip_header=True, usecols=(3,4,5,6), unpack=True )
    os.system ( cmd_to_clear )
    #############################################
    ###
    ## on phase
    nbin          = I.size
    ON            = np.zeros ( nbin, dtype=bool )
    OF            = np.ones ( nbin, dtype=bool )
    on_slice      = slice ( ran['tstart'], ran['tstop'] )
    ON[on_slice]  = True
    ## one extract at the end
    ON[on_slice.stop+1] = True
    OF[on_slice]  = False
    pulsebins     = ON.sum()
    offbins       = OF.sum()
    if args.v:
        print ("Pulse window corresponds to {:d} bins".format(pulsebins))
    ###################################
    ## ----- OFF PULSE STATISTICS -----
    I_offavg      = I[OF].mean ()
    Q_offavg      = Q[OF].mean ()
    U_offavg      = U[OF].mean ()
    V_offavg      = V[OF].mean ()

    if args.v:
        print ("INFO: Lowering baselines by ")
        print (" Delta(I) = {:.5f}".format(I_offavg))
        print (" Delta(Q) = {:.5f}".format(Q_offavg))
        print (" Delta(U) = {:.5f}".format(U_offavg))
        print (" Delta(V) = {:.5f}".format(V_offavg))

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

    if args.v:
        print ("INFO: AVG(I) = {:.5e}".format(I_offavg))
        print ("----------------")
        print ("INFO:: RMS(I) = {:.5f}".format(I_rms))
        print ("INFO:: RMS(Q) = {:.5f}".format(Q_rms))
        print ("INFO:: RMS(U) = {:.5f}".format(U_rms))
        print ("INFO:: RMS(V) = {:.5f}".format(V_rms))

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
    ###
    QoL_onavg     = np.sqrt (np.power (Q_bsl[LON]/Ltrue[LON], 2).sum () / pulsebins)
    UoL_onavg     = np.sqrt (np.power (U_bsl[LON]/Ltrue[LON], 2).sum () / pulsebins)
    ###
    dI_onavg      = I_rms / np.sqrt ( pulsebins )
    dQ_onavg      = Q_rms / np.sqrt ( pulsebins )
    dU_onavg      = U_rms / np.sqrt ( pulsebins )
    dV_onavg      = V_rms / np.sqrt ( pulsebins )

    ## Circular polarization
    V_mod_4       = np.power (V_bsl, 4) - np.power (dI_onavg, 4)
    V_mod_4[V_mod_4 < 0] = 0.0
    V_mod         = np.power ( V_mod_4, 0.25 )
    Vmask         = V_mod >= I_rms
    VON           = np.logical_and ( Vmask, ON )
    if VON.sum() <= 0:
        warnings.warn ("No Circular polarization bins", RuntimeWarning)

    V_mod_onavg   = V_mod[VON].sum() / pulsebins

    if args.v:
        print ("INFO: (S/N)_I = {:.5f}".format(I_onavg / dI_onavg))
        print ("INFO: (S/N)_L = {:.5f}".format(L_onavg / dI_onavg))
        print ("INFO: (S/N)_V_mod = {:.5f}".format(V_mod_onavg / dI_onavg))

    dL_onavg      = np.sqrt ( 
            np.power ( QoL_onavg*dQ_onavg, 2 )  + 
            np.power ( UoL_onavg*dU_onavg, 2 )
    )

    if args.v:
        print ("INFO: AVGON_Q = {:.5f} +/- {:.5f}".format (Q_onavg, dQ_onavg))
        print ("INFO: AVGON_U = {:.5f} +/- {:.5f}".format (U_onavg, dU_onavg))
        print ("INFO: AVGON_(Q/L) = {:.5f}".format(QoL_onavg))
        print ("INFO: AVGON_(U/L) = {:.5f}".format(UoL_onavg))
        print ("INFO: AVGON_I = {:.5f} +/- {:.5f} ({:d} bins)".format(I_onavg, dI_onavg, pulsebins))
        print ("INFO: AVGON_L = {:.5f} +/- {:.5f} ({:d} bins)".format(L_onavg, dL_onavg, LON.sum()))
        print ("INFO: AVGON_V = {:.5f} +/- {:.5f} ({:d} bins)".format(V_mod_onavg, dV_onavg, VON.sum()))

    polfrac_L       = L_onavg / I_onavg
    polfrac_L_err   = polfrac_L * np.sqrt ( 
            np.power (dL_onavg / L_onavg, 2)  +
            np.power (dI_onavg / I_onavg, 2) 
    )

    polfrac_V       = V_onavg / I_onavg
    polfrac_V_err   = polfrac_V * np.sqrt ( 
            np.power (dV_onavg / V_onavg, 2)  +
            np.power (dI_onavg / I_onavg, 2) 
    )

    polfrac_V_mod   = V_mod_onavg / I_onavg
    polfrac_V_mod_err = polfrac_V_mod * np.sqrt (
            np.power (dV_onavg / V_mod_onavg, 2)  +
            np.power (dI_onavg / I_onavg, 2) 
    )

    polfrac_LV      = polfrac_L + polfrac_V_mod
    polfrac_LV_err  = np.sqrt (
            np.power (polfrac_L_err, 2) +
            np.power (polfrac_V_mod_err, 2)
    )

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
    ##
    mean_ppa_deg, mean_ppa_deg_err    = weighted_mean ( ppa_deg[ppa_mask], ppa_err[ppa_mask] ) 

    # print ("---> RESULT: Linear Polarization fraction = {:.5f} +/- {:.5f}".format(polfrac_L, polfrac_L_err))
    # print ("---> RESULT: Circular Polarization fraction = {:.5f} +/- {:.5f}".format(polfrac_V, polfrac_V_err))
    # print ("---> RESULT: |Circular Polarization| fraction = {:.5f} +/- {:.5f}".format(polfrac_V_mod, polfrac_V_mod_err))
    # print ("---> RESULT: Total Polarization fraction = {:.5f} +/- {:.5f}".format(polfrac_LV, polfrac_LV_err))

    RET['lp']     = polfrac_L
    RET['vp']     = polfrac_V
    RET['vmodp']  = polfrac_V_mod
    RET['lvp']    = polfrac_LV
    RET['err_lp']     = polfrac_L_err
    RET['err_vp']     = polfrac_V_err
    RET['err_vmodp']  = polfrac_V_mod_err
    RET['err_lvp']    = polfrac_LV_err
    ## PPA
    RET['ppa_deg']          = jl ( ppa_deg[ppa_mask] )
    RET['ppa_deg_err']      = jl ( ppa_err[ppa_mask] )
    RET['ppa_deg_mask']     = jl ( ppa_mask )
    RET['mean_ppa_deg']     = mean_ppa_deg
    RET['mean_ppa_deg_err'] = mean_ppa_deg_err

    if args.v:
        print ("INFO: dI/I = {:.5f}".format (dI_onavg / I_onavg))
        print ("INFO: dL/L = {:.5f}".format (dL_onavg / L_onavg))
        print ("INFO: dV/V = {:.5f}".format (dV_onavg / V_mod_onavg))

    with open (outfile, 'w') as f:
        json.dump ( RET, f )
        # print ( json.dumps ( RET, indent=4 ) )

