"""
extension of my_pacv

- designed for linear feeds
- corrects for ionospheric RM contribution in deriving calibration solution.
- corrects for parallactic angle and position angle of the source

for linear feeds IQUV
    ionospheric RM rotates QU
    position angle rotates QU
    parallactic angle rotates UV

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

    """
    def __init__ (self, feed, freq, iquv, err_iquv, corr_angle, corr_rm):
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
        self.__freq_ghz = self.freq * 1E-3
        i, q, u, v  = iquv
        if err_iquv is None:
            err_iquv = np.zeros_like ( iquv ) + 1E-1
        ierr, qerr, uerr, verr = err_iquv
        ###################################################
        ### corrections
        ###################################################
        ### ### IONOSPHERIC RM and POSITION ANGLE CORRECTION
        ### ### PARALLACTIC ANGLE AUCH
        ## freq is in MHz
        wav2           = np.power ( C / ( self.freq * 1E6 ), 2.0 )
        qu             = (q) + (1.0j*u)
        qu_e           = (qerr) + (1.0j*uerr)
        corr_phase     = 2.0 * ( corr_angle + (corr_rm*wav2) )
        qu_corr        = qu   * np.exp ( -1.0j * corr_phase )
        qu_corr_e      = qu_e * np.exp ( -1.0j * corr_phase )
        ###
        self.icorr          = i
        self.ierr           = ierr
        self.qcorr          = np.real ( qu_corr )
        self.qerr           = np.real ( qu_corr_e )
        self.ucorr          = np.imag ( qu_corr )
        self.uerr           = np.imag ( qu_corr_e )
        self.vcorr          = v
        self.verr           = verr
        ###
        self.o_i            = i.copy()
        self.o_ierr         = ierr.copy()
        self.o_q            = q.copy()
        self.o_qerr         = qerr.copy()
        self.o_u            = u.copy()
        self.o_uerr         = uerr.copy()
        self.o_v            = v.copy()
        self.o_verr         = verr.copy()
        ###################################################
        ### get coherence products
        ###################################################
        aa             = (self.icorr + self.qcorr) * 0.5
        bb             = (self.icorr - self.qcorr) * 0.5
        ### masking necessary to determine dead channels
        if np.any(aa<0):
            aa.mask[aa<0] = True
        if np.any(bb<0):
            bb.mask[bb<0] = True
        ### error propagation 
        eaa            = self.__error_sum ( self.ierr, self.qerr, 'quadrature' )
        ebb            = self.__error_sum ( self.ierr, self.qerr, 'quadrature' )
        dab            = self.__error_sum ( eaa/aa, ebb/bb,'simple')
        ##### this pesky thingy
        ### note, would need to re define AABBCRCI in psrfits
        ### XXX
        # cr             = ucorr
        # ci             = vcorr
        cr             = self.ucorr
        ci             = self.vcorr
        ###################################################
        ### singleaxis model
        ###################################################
        self.gain       = np.sqrt ( 2.0 * np.sqrt ( aa * bb ) )
        self.g2         = np.power ( self.gain, 2.0 )
        self.gainerr    = 0.25 * self.gain * dab
        self.dgain      = 0.25 * np.log ( aa / bb )
        self.dgainerr   = 0.25 * dab 
        self.dphase     = np.arctan ( ci / cr )
        self.sigma_i    = np.mean ( np.sqrt (  (self.ucorr/self.g2)**2 + (self.vcorr/self.g2)**2 ) )
        ## i can probably remove sigma
        self.sigma      = self.sigma_i
        ###
        self.dphase_unwrap = self.__unwrap ( self.dphase )
        self.dphase_lpar_i = np.polyfit ( self.freq, self.dphase_unwrap, 1 )
        ###################################################
        ### prepare for fit
        ### lpar is the line parameters
        ### line is the actual line
        ### error is the error in the line
        ##### dphase error is wrapped
        ###################################################
        self.dphase_lpar   = np.zeros ( 2 )
        self.dphase_line   = np.zeros_like ( self.freq )
        self.dphaseerr     = np.zeros_like ( self.freq )
        ###################################################
        ### prepare yfit,yerr
        ###################################################
        self.__yfit    = np.concatenate ( (self.ucorr, self.vcorr) )
        self.__yerr    = np.concatenate ( (self.uerr, self.verr) )
        self.__yerrll  = -0.5 * np.sum ( np.log ( 2.0 * np.pi * self.__yerr**2 ) )
        ##################################################
        ### in the diag plot add the line
        self.plotxt    = f"RM_ionos = {corr_rm:.2f} angle_corr = {np.rad2deg(corr_angle):.2f} deg"

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
        dpi, bias = par
        return bias + ( dpi * self.freq * 1E-3 )

    def get_line_wrap ( self, par, hin=False):
        dpi, bias = par
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

    def model ( self, delay_pi, bias, sigma=None ):
        """ CRCI model freq in MHz """
        if sigma is None: sigma = self.sigma_i
        aa = bias + (delay_pi * self.freq * 1E-3)
        g2 = self.g2
        # g2 = 1.0
        uu = sigma * g2 * np.cos ( aa )
        vv = sigma * g2 * np.sin ( aa )
        return uu,vv

    def __un_solver__ ( self, DIR ):
        """ performs minimization using ultranest
            
            line ~ bias + delay*freq

            bias in between 0.--> 2pi
            pi*delay in -30ns to 30ns
            delay in -30ns/pi to 30ns/pi
            pi*delay in -4ns to 4ns

            freq is in GHz
            
            par --> [delay, bias]
        """
        import ultranest
        import ultranest.stepsampler

        ##
        SLICE_DPI   = 0
        SLICE_BIAS  = 1
        names  = ['DELAY_PI', 'BIAS']
        ##
        def priorer (cube):
            param = np.zeros_like ( cube )
            param[SLICE_DPI]    = (-400.0) + ( 800.0 * cube[SLICE_DPI] )
            param[SLICE_BIAS]   =  2.0 * np.pi * cube[SLICE_BIAS] 
            ## have BIAS in [0., 2.0*np.pi)
            # param[SLICE_BIAS]   = ( -1.5 * np.pi ) +  ( 3.0 * np.pi * cube[SLICE_BIAS]  )
            return param
        def logll ( par ):
            yy   = np.concatenate ( self.model (*par)  )
            return -0.5 * np.sum ( np.power ( ( yy - self.__yfit ) / self.__yerr, 2.0 ) ) + self.__yerrll

        sampler             = ultranest.ReactiveNestedSampler (
            names,
            logll, priorer,
            wrapped_params = [False, True],
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
            isol,isol_err       = [121.31,1.19], [1e-2, 1e-2]
        else:
            isol,isol_err       = self.__un_solver__ ( dir )
        ###
        self.dphase_lpar[:] = isol[:2]
        self.dphaseerr[:]   = self.__wrap (
            isol_err[1] + ( isol_err[0]*self.freq*1E-3 )
        )
        # self.sigma          = isol[2]
        self.sigma          = self.sigma_i
        ## history
        delay           = isol[0] / np.pi
        bias            = isol[1]
        self.history   += f"Fitted dphase\n\tbias={bias:.3f} delay={delay:.3f} us\n"

    def diag_plot ( self, save=None ):
        """  plots a diagnostic plot """
        mu, mv      = self.model ( *self.dphase_lpar, self.sigma )
        iu, iv      = self.model ( *self.dphase_lpar_i, self.sigma  )
        ##########################################
        import matplotlib.pyplot as plt
        
        if save is None:
            fig         = plt.figure ()
        else:
            fig         = plt.figure (dpi=300, figsize=(7,5))
        ax,qx,ux,vx    = fig.subplots ( 4,1,sharex=True )

        ax.scatter ( self.freq, self.dphase, marker='.',c='k', label='DATA' )
        ax.plot ( self.freq, self.get_pval (self.dphase_lpar_i), ls='-',c='r',label='INITIAL' )
        ax.plot ( self.freq, self.get_line_wrap(self.dphase_lpar), ls='-',c='b',label='FIT' )
        ax.set_ylabel ('DPHASE / rad')
        ax.legend (loc='best')

        MS=1
        LW=0.5

        qx.errorbar ( self.freq, self.o_q, yerr=self.o_qerr, c='k', label='DATA', alpha=0.4, marker='o', markersize=MS, linewidth=LW )
        qx.errorbar ( self.freq, self.qcorr, yerr=np.abs( self.qerr ), c='k', label='DATA-FIT', marker='o', markersize=MS, linewidth=LW )
        # qx.plot ( self.freq, mq, c='b', label='FIT' )
        # qx.plot ( self.freq, iq, c='r', label='INITIAL' )
        qx.set_ylabel ('Q')
        qx.legend (loc='best')

        ux.errorbar ( self.freq, self.o_u, yerr=self.o_uerr, c='k', label='DATA', alpha=0.4, marker='o', markersize=MS, linewidth=LW )
        ux.errorbar ( self.freq, self.ucorr, yerr=np.abs( self.uerr ), c='r', label='DATA-FIT', marker='o', markersize=MS, linewidth=LW )

        ux.plot ( self.freq, mu, c='b', label='FIT' )
        # ux.plot ( self.freq, iu, c='r', label='INITIAL' )
        ux.set_ylabel ('U')
        ux.legend (loc='best')

        vx.errorbar ( self.freq, self.o_v, yerr=self.o_verr, c='k', label='DATA', alpha=0.4, marker='o', markersize=MS, linewidth=LW )
        vx.errorbar ( self.freq, self.vcorr, yerr=np.abs( self.verr ), c='r', label='DATA-FIT', marker='o', markersize=MS, linewidth=LW )

        vx.plot ( self.freq, mv, c='b', label='FIT' )
        # ux.plot ( self.freq, iu, c='r', label='INITIAL' )
        vx.set_ylabel ('V')
        vx.legend (loc='best')

        vx.set_xlabel ('Freq / MHz')
        # fig.suptitle (f"SIGMA = {caler.sigma_i:.3f} --> {caler.sigma:.3f}")
        fig.suptitle ( self.plotxt )

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
        dpx.errorbar ( self.freq, self.dphase, yerr=self.dphaseerr, ls='', marker='s', markersize=2, capsize=2, color='b' )


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
