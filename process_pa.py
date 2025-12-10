"""

takes in :make_csv.py: output and performs par corrections

maybe add it to :make_csv.py: to make corrections OTF


"""

import os
import numpy as np
import pandas as pd

import astropy.time as at
import astropy.coordinates as asc
import astropy.units as au

###############################
C     = 299792458.0 # m/s
gmrt  = asc.EarthLocation.from_geocentric (1656342.30, 5797947.77, 2073243.16, unit="m")
R3_sc = asc.SkyCoord ( 29.50312583 * au.degree, 65.71675422 * au.degree, frame='icrs' )
###############################
class MountDirectional:
    """
    What psrchive does to position angles!

    make it modular

    """
    BASIS_0 = np.array ( [1.0, 0.0, 0.0], dtype=np.float64 )
    BASIS_1 = np.array ( [0.0, 1.0, 0.0], dtype=np.float64 )
    BASIS_2 = np.array ( [0.0, 0.0, 1.0], dtype=np.float64 )
    def __init__ (self, skypos, obs, v=False):
        """
        skypos: SkyCoord
        obs: astropy.coordinates.EarthLocation

        """
        self.v  = v


        ra   = skypos.ra.radian
        dec  = skypos.dec.radian
        self.skypos = skypos

        if self.v:
            print (f" ra/dec = {np.rad2deg ([ra, dec])}")

        ## source basis
        self.source_basis = self.__transpose__ ( np.dot (
                self.__rotation__ ( MountDirectional.BASIS_0, ra )  ,
                self.__rotation__ ( MountDirectional.BASIS_1, -dec) 
            )
        )

        # lat, long= 0.0,0.0
        # obs  = asc.EarthLocation ( 0 * au.degree, 0 * au.degree )

        self.obs  = obs
        self.lat  = obs.lat.to(au.radian).value
        self.long = obs.lon.to(au.radian).value

    def __call__ (self, obstime):
        """
        obstime: astropy.Time 

        return radian of angle correction term
        """
        otime = at.Time ( obstime, location=self.obs )
        lst   = otime.sidereal_time ('mean').to(au.radian).value

        if self.v:
            print (f" lat/long/lst  = {np.rad2deg ([lat, long, lst])}")

        ## obs basis
        obs_basis    = self.__transpose__ ( np.dot (
            self.__rotation__ ( MountDirectional.BASIS_0, lst )  ,
            self.__rotation__ ( MountDirectional.BASIS_1, -self.lat ) 
            )
        )

        ## from source
        from_source  = np.dot ( obs_basis, self.source_basis[2] )

        ## get azimuth, zenith
        aa                = asc.AltAz ( obstime=obstime, location=self.obs )
        saa               = self.skypos.transform_to ( aa )

        alt               = saa.alt.to(au.radian).value
        azimuth           = saa.az.to(au.radian).value
        zenith            = (0.5*np.pi) - alt

        if args.v:
            print (f" alt,az,zenith = ", np.rad2deg ( [alt, azimuth, zenith] ))

        ## 
        R1                = self.__rotation__ ( MountDirectional.BASIS_2, np.pi - azimuth )
        R2                = self.__rotation__ ( MountDirectional.BASIS_1, -zenith )
        basis             = np.dot ( R2, R1 )

        ## north
        north        = np.linalg.multi_dot ( ( basis, obs_basis, self.source_basis[0] ) )
        vertical     = - np.arctan2 ( north[1], north[0] )

        if args.v:
            print (f" MJD={obstime.mjd:.6f} vertical={np.rad2deg(self.vertical):.3f} deg")
        return vertical

    def __transpose__ (self, t):
        """ transpose """
        return np.transpose ( t )

    def __rotation__ (self, rotaxis, theta):
        """
        use the Rodriges formula 

        this is in matrix notation
        """
        kx,ky,kz  = rotaxis
        st        = np.sin ( theta )
        ct        = np.cos ( theta )
        ##
        K = np.array ( [ [0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0] ] ) 
        K2 = np.dot ( K, K )
        ##
        R    = np.eye ( 3 ) + (st * K) + ( (1.0 - ct)*K2 )
        return R
###############################
def get_args ():
    import argparse as agp
    ap   = agp.ArgumentParser ('process_pa', description='Measures angle correction', epilog='Part of GMRT/FRB')
    add  = ap.add_argument
    add ('npz', help='npz files', nargs='+')
    add ('-v', '--verbose', help='Verbose', dest='v', action='store_true')
    return ap.parse_args ()

if __name__ == "__main__":
    args = get_args ()
    ##
    DD   = {'mjd':[], }
    DD['pa_org'] = []
    DD['paerr_org'] = []
    DD['fref']  = []
    DD['vertical'] = []
    DD['rmlc'] = []
    DD['rmld'] = []
    DD['rm']  = []
    DD['rmerr'] = []
    ##
    md      = MountDirectional ( R3_sc, gmrt )
    # read file
    for l in args.npz:
        f = np.load ( l )
        ###
        il = os.path.basename ( l )
        mjd   = float ( il.split('_')[0] )
        DD['mjd'].append ( mjd )
        ###
        DD['pa_org'].append ( f['pa_qu'] )
        DD['paerr_org'].append ( f['paerr_qu'] )
        ##
        freq = f['freq_list']
        fref = np.median ( freq )
        DD['fref'].append ( fref )
        DD['rm'].append ( f['rm_qu'] )
        DD['rmerr'].append ( f['rmerr_qu'] )
        ##
        wref = C / fref / 1E6
        wref_d = C / 650 / 1E6
        rmlc   = f['rm_qu'] * ( wref**2 )
        rmld   = f['rm_qu'] * ( wref_d**2 )
        DD['rmlc'].append ( np.rad2deg (np.mod(rmlc, np.pi) ) )
        DD['rmld'].append ( np.rad2deg (np.mod(rmld, np.pi) ) )
        ##
        vertical = md ( at.Time ( mjd, format='mjd', scale='utc' ) )
        DD['vertical'].append ( np.rad2deg ( vertical ) )
    ##
    df = pd.DataFrame ( DD )
    print (df.head())
    df.to_csv('band4_godsplan_processpa_df.csv', index=False)
