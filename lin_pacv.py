"""
extension of my_pacv

- designed for linear feeds
- corrects for ionospheric RM contribution in deriving calibration solution.
- corrects for parallactic angle and position angle of the source

for linear feeds IQUV
    ionospheric RM rotates QU
    position angle rotates QU
    parallactic angle rotates QU
    delay rotates UV
    dgain boosts IQ

Mueller matrix

BOOST[IQ] * ROTATION[UV] * ROTATION[QU]
^ DGAIN ^ -  ^ DELAY ^   -  ^ PAR-ANGLE & IONOS-RM & ISM-RM ^
^   X   ^ -  ^   D   ^   -  ^ PHI   ^

see :measure_rm_pa_delay_un_equad.py:

IQUV = [
    COSH(X) + LP*SINH(X)*COS(PHI),
    SINH(X) + LP*COSH(X)*COS(PHI),
    LP*SIN(PHI)*COS(D),
    LP*SIN(PHI)*SIN(D),
]

this is so complicated
"""
import datetime

import numpy as np

C       = 299792458.0 # m/s

__all__ = ['read_pkl', 'MyPACV']

def read_pkl (file):
    """ return dict(fbw, nchan, fcen, mjd, source), freq, sc """
    import pickle as pkl
    with open (file, "rb") as f:
        k  = pkl.load ( f, encoding='latin1' )
    sc = np.ma.array (k['data'][0], mask=k['wts'][0], fill_value=np.nan)
    f  = np.linspace ( -0.5 * k['fbw'], 0.5 * k['fbw'], k['nchan'] ) + k['fcen']
    wt = np.array (k['wts'][0].sum ( (0,2) ), dtype=bool)
    freq = np.ma.array ( f, mask=wt, fill_value=np.nan )
    return dict(fbw=k['fbw'], fcen=k['fcen'], mjd=k['mjd'], source=k['src'], nchan=k['nchan'],basis=k['basis']), freq, sc

class MyPACV:
    """
    made with linear feeds

    """
    def __init__ (self, feed, freq, iquv, err_iquv):
        """

        feed should be "LIN"
        freq should be in MHz

        iquv is ON-OFF

        err_iquv is std-dev of OFF iquv

        pal_angle is the parallactic_angle in radians
        ionosrm is the ionospheric RM contribution
        pos_angle is the position angle of the source
        both of them are corrected for before deriving solution

        Ionospheric RM, position angle correct QU
        then,
        parallactic angle correct QU
        (the U you use here is RM-pos corrected)
        then,
        using the corrected IQ you get GAIN,DGAIN

        i thought parallactic angle would rotate UV but it actually 
        rotates QU 
        """
        self.history = f"Created at {datetime.datetime.utcnow().isoformat()}\n"
        ###################################################
        ### LOAD logic
        ###################################################
        self.feed    = feed.upper ()
        if self.feed not in ["LIN"]:
            raise ValueError ("Feed not recognized, feed = ", self.feed)
        self.freq    = freq.copy ()
        self.nchan   = self.freq.size
        self.wav2    = np.power ( C / ( self.freq * 1E6 ), 2.0 )
        ## centering wav2
        self.wav2    -= np.power ( C / ( self.freq[self.nchan//2] * 1E6 ), 2.0 )
        self.__freq_ghz = self.freq * 1E-3
        self.i, self.q, self.u, self.v  = iquv
        if err_iquv is None:
            err_iquv = np.zeros_like ( iquv ) + 1E-1
        self.ierr, self.qerr, self.uerr, self.verr = err_iquv
        ###################################################
        ### get coherence products
        ###################################################
        aa             = (self.i + self.q) * 0.5
        bb             = (self.i - self.q) * 0.5
        ### masking necessary to determine dead channels
        if np.any(aa<0):
            aa.mask[aa<0] = True
        if np.any(bb<0):
            bb.mask[bb<0] = True
        ### error propagation 
        eaa    = self.__error_sum ( self.ierr, self.qerr, 'quadrature' )
        ebb    = self.__error_sum ( self.ierr, self.qerr, 'quadrature' )
        dab    = self.__error_sum ( eaa/aa, ebb/bb,'simple')
        ###################################################
        ### singleaxis model
        ###################################################
        self.gain       = np.sqrt ( 2.0 * np.sqrt ( aa * bb ) )
        self.g2         = np.power ( self.gain, 2.0 )
        self.gainerr    = 0.25 * self.gain * dab
        self.dgain      = 0.25 * np.log ( aa / bb )
        self.dgainerr   = 0.25 * dab 
        ###################################################
        ### prepare for fit
        ### lpar is the line parameters
        ### line is the actual line
        ### error is the error in the line
        ##### dphase error is wrapped
        ###################################################
        self.dphase_lpar   = np.zeros ( 2 )
        self.dphase        = np.zeros_like ( self.freq )
        self.dphaseerr     = np.zeros_like ( self.freq )
        ###################################################
        ### prepare yfit,yerr
        ###################################################
        self.__yfit    = np.concatenate ( 
            (self.i, self.q, self.u, self.v) 
        )
        self.__yerr    = np.concatenate ( 
            (self.ierr, self.qerr, self.uerr, self.verr) 
        )
        self.__yerrll  = -0.5 * np.sum ( np.log ( 2.0 * np.pi * self.__yerr**2 ) )
        ##################################################
        ### in the diag plot add the line
        # self.plotxt    = f"RM_ionos = {corr_rm:.2f} angle_corr = {np.rad2deg(corr_angle):.2f} deg"
        self.plotxt    = ""

    def __error_sum (self, dx, dy, method):
        """ either sum in quadrature or simple """
        ## quadrature
        if method == 'quadrature':
            return np.sqrt ( dx**2 + dy**2 )
        elif method == 'rms':
            return np.sqrt ( dx**1 + dy**1 )
        elif method == 'simple':
            return dx + dy
        else:
            raise ValueError ("Method not understood")

    def __wrap (self, a):
        """ wrap """
        return np.arctan (np.tan(a))

    def get_pval (self, par):
        return self.__wrap ( np.polyval ( par, self.freq ) )

    def get_line ( self, par ):
        """ do i need this """
        raise RuntimeError (" still using this?")
        rm, delay_pi, pa, bias, lp = par
        return bias + ( dpi * self.freq * 1E-3 )

    def get_line_wrap ( self, par, hin=False):
        """ do i need this """
        raise RuntimeError (" still using this?")
        rm, delay_pi, pa, bias, lp = par
        aa        = bias + ( dpi * self.freq * 1E-3 )
        if hin:
            return self.__wrap ( -0.5 * aa )
        else:
            return self.__wrap ( aa )
    
    def __unwrap (self, a):
        """ to unwrap """
        # return np.unwrap (a, period=np.pi, discont=np.pi)
        return np.unwrap (a, period=np.pi)

    def __str__ (self):
        """ goes to history """
        return self.history

    def model ( self, rm, delay_pi, pa, bias, lp ):
        """ 
        rm, delay_pi, pa, bias, lp
        """
        ## angles
        X   = 2.0 * self.dgain
        phi = 2.0 * ( (self.wav2 * rm) + pa )
        D   = bias + (delay_pi * self.freq * 1E-3)
        g2 = self.g2

        ## forward model
        ii = g2 * ( 
            np.cosh ( X ) + ( lp * np.sinh ( X ) * np.cos ( phi ) ) 
        )
        qq = g2 * ( 
            np.sinh ( X ) + ( lp * np.cosh ( X ) * np.cos ( phi ) ) 
        )
        uu = g2 * lp * np.sin ( phi ) * np.cos ( D )
        vv = g2 * lp * np.sin ( phi ) * np.sin ( D )
        ## return
        return ii,qq,uu,vv

    def __un_solver__ ( self, DIR ):
        """ performs minimization using ultranest

            rm, delay_pi, pa, bias, lp
            
            bias in between 0.--> 2pi
            pi*delay in -30ns to 30ns

            freq is in GHz
            
            par --> [delay, bias]
        """
        import ultranest
        import ultranest.stepsampler

        ##
        SLICE_RM    = 0
        SLICE_DPI   = 1
        SLICE_PA    = 2
        SLICE_BIAS  = 3
        SLICE_LP    = 4
        names  = [
            'RM', 'DELAY_PI', 'PA', 'BIAS', 'LP'
        ]
        ##
        def priorer (cube):
            param = np.zeros_like ( cube )
            param[SLICE_RM]    = (-400.0) + ( 800.0 * cube[SLICE_RM] )
            param[SLICE_DPI]   = (-400.0) + ( 800.0 * cube[SLICE_DPI] )
            param[SLICE_PA]    =  1.0 * np.pi * cube[SLICE_PA] 
            param[SLICE_BIAS]  =  2.0 * np.pi * cube[SLICE_BIAS] 
            param[SLICE_LP]    =  1.0 * cube[SLICE_LP] 
            return param

        def logll ( par ):
            yy   = np.concatenate ( self.model (*par)  )
            return -0.5 * np.sum ( np.power ( ( yy - self.__yfit ) / self.__yerr, 2.0 ) ) + self.__yerrll

        sampler             = ultranest.ReactiveNestedSampler (
            names,
            logll, priorer,
            wrapped_params = [False, False, True, True, False],
            num_test_samples = 100,
            draw_multiple = True,
            num_bootstraps = 100,
            log_dir = DIR
        )
        sampler.stepsampler = ultranest.stepsampler.SliceSampler (
            nsteps = 25,
            generate_direction = ultranest.stepsampler.generate_cube_oriented_differential_direction,
            adaptive_nsteps='move-distance',
        )
        result              = sampler.run (
            min_num_live_points = 1024,
            frac_remain = 1E-4,
            min_ess = 512,
        )
        sampler.plot_corner ()
        ###
        popt          = result['posterior']['median']
        perr          = result['posterior']['stdev']
        ###
        return popt, perr

    def fit_dphase (self, dir="my_pacv", test=False):
        """

        fit a straight line
        """
        if test:
            isol       = [-121.31,32, np.pi*0.25, np.pi/8, 0.8]
            isol_err   = [1E-2] * 5
        else:
            isol,isol_err       = self.__un_solver__ ( dir )
        ###
        rm, delay_pi, pa, bias, lp = isol
        erm, edelay_pi, epa, ebias, elp = isol_err
        ###
        self.isol     = isol
        self.isol_err = isol_err
        ###
        self.dphase_lpar[:] = delay_pi, bias
        self.dphase[:]      = self.__wrap (
            -0.5 * ( bias + ( delay_pi * self.freq * 1E-3 ) )
        )
        self.dphaseerr[:]   = self.__wrap (
            -0.5 * ( ebias + ( edelay_pi * self.freq * 1E-3 ) )
        )
        ## history
        delay           = delay_pi / np.pi
        self.history   += f"Fitted dphase\n\tbias={bias:.3f} delay={delay:.3f} us\n"

    def diag_plot ( self, save=None ):
        """  plots a diagnostic plot """
        rm, delay_pi, pa, bias, lp = self.isol
        ii,qq,uu,vv = self.model ( rm, delay_pi, pa, bias, lp )
        ##########################################
        import matplotlib.pyplot as plt
        
        if save is None:
            fig         = plt.figure ()
        else:
            fig         = plt.figure (dpi=300, figsize=(7,5))
        ix,qx,ux,vx     = fig.subplots ( 4,1,sharex=True )

        MS=1
        LW=0.5

        ix.errorbar ( self.freq, self.i/self.g2, yerr=self.ierr/self.g2, c='k', label='DATA', alpha=0.4, marker='o', markersize=MS, linewidth=LW )
        ix.plot ( self.freq, ii/self.g2, c='cyan', label='FIT', marker='o', markersize=MS, linewidth=LW )
        ix.set_ylabel ('I/G2')
        ix.legend (loc='best')

        qx.errorbar ( self.freq, self.q/self.g2, yerr=self.qerr/self.g2, c='k', label='DATA', alpha=0.4, marker='o', markersize=MS, linewidth=LW )
        qx.plot ( self.freq, qq/self.g2, c='r', label='FIT', marker='o', markersize=MS, linewidth=LW )
        qx.set_ylabel ('Q/G2')
        qx.legend (loc='best')

        ux.errorbar ( self.freq, self.u/self.g2, yerr=self.uerr/self.g2, c='k', label='DATA', alpha=0.4, marker='o', markersize=MS, linewidth=LW )
        ux.plot ( self.freq, uu/self.g2, c='b', label='FIT', marker='o', markersize=MS, linewidth=LW )
        ux.set_ylabel ('U/G2')
        ux.legend (loc='best')

        vx.errorbar ( self.freq, self.v/self.g2, yerr=self.verr/self.g2, c='k', label='DATA', alpha=0.4, marker='o', markersize=MS, linewidth=LW )
        vx.plot ( self.freq, vv/self.g2, c='g', label='FIT', marker='o', markersize=MS, linewidth=LW )
        vx.set_ylabel ('V/G2')
        vx.legend (loc='best')

        vx.set_xlabel ('Freq / MHz')

        rm, delay_pi, pa, bias, lp = self.isol
        erm, edelay_pi, epa, ebias, elp = self.isol_err
        fig.suptitle (
            f"RM = {rm:.2f}+/-{erm:.2f} Delay={delay_pi/np.pi:.1f}+-{edelay_pi/np.pi:.1f} ns\n"+
            f"LP = {lp:.1f}+/-{elp:.1f}"
        )
        # fig.suptitle ( self.plotxt )

        if save is None:
            plt.show ()
        else:
            fig.savefig (save, dpi=300, bbox_inches='tight')

    def sol_plot ( self, save=None ):
        """  plots a solution plot """
        import matplotlib.pyplot as plt
        
        if save is None:
            fig         = plt.figure ()
        else:
            fig         = plt.figure (dpi=300, figsize=(7,5))
        gx,dgx,dpx      = fig.subplots ( 3,1,sharex=True )

        gx.errorbar ( self.freq, self.gain, yerr=self.gainerr, ls='', marker='s', markersize=2, capsize=2, color='b' )
        dgx.errorbar ( self.freq, self.dgain, yerr=self.dgainerr, ls='', marker='s', markersize=2, capsize=2, color='b' )
        dpx.errorbar ( self.freq, self.dphase, yerr=np.abs( self.dphaseerr ), ls='', marker='s', markersize=2, capsize=2, color='b' )


        gx.set_ylabel ('Gain')
        dgx.set_ylabel ('Diff\nGain')
        dpx.set_ylabel ('Diff\nPhase')

        dpx.set_xlabel ('Freq / MHz')

        if save is None:
            plt.show ()
        else:
            fig.savefig (save, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    #######################
    FILE_UNCAL  = "3C138_bm3_pa_550_200_32_09dec2020.raw.calonoff.ar.T32_npy.pkl"
    FILE_UNCAL  = "3C138_bm3_pa_550_200_32_09dec2020.raw.calonoff.ar.T_npy.pkl"
    FILE_PACV   = "3C138_bm3_pa_550_200_32_09dec2020.raw.calonoff.ar.T32.mypacv"
    #######################
    freq, sc    = read_pkl ( FILE_UNCAL )
    pp          = sc[0].mean(0)
    ## ON-phase is more than 60% of the maximum
    mask        = pp >= (0.60 * pp.max())
    ff          = sc[...,mask].mean(-1) - sc[...,~mask].mean(-1)
    #######################
    caler       = MyPACV ( "CIRC", freq, ff )
    caler.fit_dphase (test=True)
    print ( caler )
    #######################
    caler.diag_plot ()
    #######################
