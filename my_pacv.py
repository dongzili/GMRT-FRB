
import datetime

import numpy as np

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
    def __init__ (self, feed, freq, iquv, err_iquv):
        """

        feed should be ["LIN", "CIRC"]
        freq should be in MHz

        iquv is ON-OFF

        err_iquv is std-dev of OFF iquv


        """
        self.history = f"Created at {datetime.datetime.utcnow().isoformat()}\n"
        ##############
        self.feed    = feed.upper ()
        if self.feed not in ["LIN", "CIRC"]:
            raise ValueError ("Feed not recognized, feed = ", self.feed)
        self.freq    = freq.copy ()
        self.__freq_ghz = self.freq * 1E-3
        self.i, self.q, self.u, self.v  = iquv
        if err_iquv is None:
            err_iquv = np.zeros_like ( iquv ) + 1E-1
        self.ierr, self.qerr, self.uerr, self.verr = err_iquv

        ###
        self.g2      = np.zeros_like ( self.freq )
        self.gain    = np.zeros_like ( self.freq )
        self.dgain   = np.zeros_like ( self.freq )
        self.dphase  = np.zeros_like ( self.freq )
        self.dphase_unwrap = np.zeros_like ( self.freq )
        self.sigma   = np.sqrt ( self.q**2 + self.u**2 ).mean()
        ##
        # errors
        self.gainerr   = np.zeros_like ( self.freq )
        self.dgainerr  = np.zeros_like ( self.freq )
        self.dphaseerr = np.zeros_like ( self.freq )
        ## dphase error is wrapped
        ##
        self.dphase_line   = np.zeros_like ( self.freq )
        self.dphase_lpar   = np.zeros ( 2 )
        self.dphase_lpar_i = np.zeros ( 2 )
        ###
        self.__yfit    = None
        self.__yerr    = None
        if self.feed == "LIN":
            self.initialize_linear ()
        if self.feed == "CIRC":
            self.initialize_circular ()
        ###
        self.initialize_dphase ()
        self.sigma_i       = np.mean ( np.sqrt (  (self.q/self.g2)**2 + (self.u/self.g2)**2 ) )
        self.sigma         = None

    def initialize_linear (self):
        """ 
            initialize 

            based on Britton(2000)
        """  
        if self.feed != "LIN":
            raise ValueError("Feed mismatch")
        aa             = (self.i + self.q) * 0.5
        bb             = (self.i - self.q) * 0.5
        eaa            = np.sqrt ( (self.ierr)**2 + (self.verr)**2 )
        ebb            = np.sqrt ( (self.ierr)**2 + (self.verr)**2 )
        dab            = np.sqrt (  (eaa/aa)**2 + (ebb/bb)**2 )
        ##### this pesky thingy
        ### note, would need to re define AABBCRCI in psrfits
        ### XXX
        cr             = self.u
        ci             = self.v
        #####
        self.gain[:]       = np.sqrt ( 2.0 * np.sqrt ( aa * bb ) )
        self.gainerr[:]    = 0.25 * self.gain * dab
        self.dgain[:]      = 0.25 * np.log ( aa / bb )
        self.dgainerr[:]   = 0.25 * dab / ( aa / bb )
        self.dphase[:]     = np.arctan ( ci / cr )
        self.g2[:]         = self.gain[:]**2
        #####
        self.__yfit    = np.concatenate ( (self.u, self.v) )
        self.__yerr    = np.concatenate ( (self.uerr, self.verr) )
        ####
        self.history   += "Solved for LINEAR feeds\n"

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

    def initialize_circular (self):
        """ 
            initialize 

            based on Britton(2000)

            sum in quadrature
            DELTA ( A * B ) = sqrt ( (DELTA(A)/A)**2 + (DELTA(B)/B)**2 )

            Error propagation
            DELTA(G) / G = 0.25 * DELTA ( A * B )
            DELTA(GAMMA)  = 0.25 * DELTA ( A / B ) / ( A / B )

            WTF why are not my errors matching?

        """  
        if self.feed != "CIRC":
            raise ValueError("Feed mismatch")
        aa             = (self.i + self.v) * 0.5
        bb             = (self.i - self.v) * 0.5
        eaa            = self.__error_sum ( self.ierr, self.verr, 'quadrature' )
        ebb            = self.__error_sum ( self.ierr, self.verr, 'quadrature' )
        dab            = self.__error_sum ( eaa/aa, ebb/bb, 'simple' )
        ##### this pesky thingy
        ### note, would need to re define AABBCRCI in psrfits
        ### XXX
        # cr             = u
        # ci             = q
        cr             = self.q
        ci             = self.u
        ## CRCI
        self.__yfit    = np.concatenate ( (self.q, self.u) )
        self.__yerr    = np.concatenate ( (self.qerr, self.uerr) )
        self.__yerrll  = -0.5 * np.sum ( np.log ( 2.0 * np.pi * self.__yerr**2 ) )
        #####
        self.gain[:]       = np.sqrt ( 2.0 * np.sqrt ( aa * bb ) )
        self.gainerr[:]    = 0.25 * self.gain * dab
        self.dgain[:]      = 0.25 * np.log ( aa / bb )
        self.dgainerr[:]   = 0.25 * dab 
        self.dphase[:]     = np.arctan ( ci / cr )
        self.g2[:]         = self.gain[:]**2
        #####
        ## error
        #####
        self.history   += "Solved for CIRCULAR feeds\n"

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

    def initialize_dphase (self):
        """ for circular feeds """
        self.dphase_unwrap[:] = self.__unwrap ( self.dphase )
        isol                  = np.polyfit ( self.freq, self.dphase_unwrap, 1 )
        self.dphase_lpar_i[:] = isol
        # self.history   += f"Initial dphase\n\t{isol[1]:.3f} + {isol[0]:.3f}*freq\n"
        delay           = isol[0] * 1E3 / np.pi
        bias            = isol[1]
        self.history   += f"Initial dphase\n\tbias={bias:.3f} delay={delay:.3f} us\n"

    def __str__ (self):
        """ goes to history """
        return self.history

    def model ( self, delay_pi, bias, sigma=None ):
        """ CRCI model freq in MHz """
        if sigma is None: sigma = self.sigma_i
        aa = bias + (delay_pi * self.freq * 1E-3)
        g2 = self.g2
        # g2 = 1.0
        qq = sigma * g2 * np.cos ( aa )
        uu = sigma * g2 * np.sin ( aa )
        return qq, uu

    def __solver__ ( self ):
        """ performs minimization using scipy.optimize 
            
            line ~ bias + delay*freq

            bias in between 0.--> 2pi
            pi*delay in -30ns to 30ns
            delay in -30ns/pi to 30ns/pi
            sigma 

            freq is in GHz
            
            par --> [delay, bias, sigma]
        """
        import scipy.optimize as so
        ## curve_fit function
        def foo ( f, dp, b): return np.concatenate ( self.model ( dp, b ) )
        names  = ['DELAY_PI', 'BIAS']
        bounds = so.Bounds ( [ -400.0, -2.0*np.pi], [ 400.0, 2.0*np.pi] )
        p0     = self.dphase_lpar_i.copy()
        p0[1]  = np.mod ( p0[1], 2.0*np.pi )
        ###
        popt, pconv = so.curve_fit ( foo, self.freq, self.__yfit, p0=p0, bounds=bounds, max_nfev=100000 )
        perr   = np.sqrt ( np.diag ( pconv ) )
        ###
        return popt, perr

    def __test_solver__ ( self, DIR="my_pacv" ):
        # popt  = [101.790, 2.8242]
        popt  = [84.48, 3.46]
        perr  = [1e-2, 1e-2]
        return popt, perr

    def __un_solver_2__ ( self, DIR ):
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

    def __un_solver__ ( self, DIR):
        """ performs minimization using ultranest
            
            line ~ bias + delay*freq

            bias in between 0.--> 2pi
            pi*delay in -30ns to 30ns
            delay in -30ns/pi to 30ns/pi
            pi*delay in -4ns to 4ns
            sigma 

            freq is in GHz
            
            par --> [delay, bias, sigma]
        """
        import ultranest
        import ultranest.stepsampler
        ##
        SLICE_DPI   = 0
        SLICE_BIAS  = 1
        SLICE_SIGMA = 2
        names  = ['DELAY_PI', 'BIAS', 'SIGMA']
        ##
        def priorer (cube):
            param = np.zeros_like ( cube )
            param[SLICE_DPI]    = (-8.0) + ( 16.0 * cube[SLICE_DPI] )
            param[SLICE_BIAS]   = 2.0 * np.pi * cube[SLICE_BIAS] 
            param[SLICE_SIGMA]  = cube[SLICE_SIGMA]
            return param
        def logll ( par ):
            yy   = np.concatenate ( self.model (*par)  )
            return -0.5 * np.sum ( np.power ( yy - self.__yfit, 2.0 ) )

        sampler             = ultranest.ReactiveNestedSampler (
            names,
            logll, priorer,
            wrapped_params = [False, True, False],
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
            isol,isol_err       = self.__test_solver__ ()
        else:
            # isol,isol_err       = self.__solver__ ()
            # isol,isol_err       = self.__un_solver__ ()
            isol,isol_err       = self.__un_solver_2__ ( dir )
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
        mq, mu      = self.model ( *self.dphase_lpar, self.sigma )
        iq, iu      = self.model ( *self.dphase_lpar_i, self.sigma  )
        ##########################################
        import matplotlib.pyplot as plt
        
        if save is None:
            fig         = plt.figure ()
        else:
            fig         = plt.figure (dpi=300, figsize=(7,5))
        ax,qx,ux    = fig.subplots ( 3,1,sharex=True )

        # ax.scatter ( freq, caler.dphase_unwrap, marker='.',c='k', label='DATA' )
        # ax.plot ( freq, caler.get_pval(caler.dphase_lpar_i), ls='-',c='r',label='INITIAL' )
        # ax.plot ( freq, caler.get_line(caler.dphase_lpar), ls='-',c='b',label='FIT' )

        ax.scatter ( self.freq, self.dphase, marker='.',c='k', label='DATA' )
        ax.plot ( self.freq, self.get_pval (self.dphase_lpar_i), ls='-',c='r',label='INITIAL' )
        ax.plot ( self.freq, self.get_line_wrap(self.dphase_lpar), ls='-',c='b',label='FIT' )

        ax.set_ylabel ('DPHASE / rad')
        ax.legend (loc='best')

        qx.plot ( self.freq, self.q, c='k', label='DATA' )
        qx.plot ( self.freq, mq, c='b', label='FIT' )
        # qx.plot ( self.freq, iq, c='r', label='INITIAL' )
        qx.set_ylabel ('Q')
        qx.legend (loc='best')

        ux.plot ( self.freq, self.u, c='k', label='DATA' )
        ux.plot ( self.freq, mu, c='b', label='FIT' )
        # ux.plot ( self.freq, iu, c='r', label='INITIAL' )
        ux.set_ylabel ('U')
        ux.legend (loc='best')

        ux.set_xlabel ('Freq / MHz')
        # fig.suptitle (f"SIGMA = {caler.sigma_i:.3f} --> {caler.sigma:.3f}")

        if save is None:
            plt.show ()
        else:
            fig.savefig (save, dpi=300, bbox_inches='tight')

    def sol_plot ( self, save=None ):
        """  plots a solution plot """
        mq, mu      = self.model ( *self.dphase_lpar, self.sigma )
        iq, iu      = self.model ( *self.dphase_lpar_i, self.sigma  )
        ##########################################
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
