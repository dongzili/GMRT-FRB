
import numpy as np


class Redigitize:
    """
    Class performing redigitization

    uGMRT filterbank is 16bit. 
    It carries the bandshape and all. 
    Need to whiten it
    """
    def __init__ (self, gulp, nchans, npol, odtype=np.uint8):
        """
        Args
            gulp: int
            nchans: int
                Sets the work array size
        """
        ## output dtype
        if odtype not in (np.uint8, np.uint16):
            raise ValueError ("output dtype not supported")
        self.odtype     = odtype
        iinfo           = np.iinfo (self.odtype)
        ran             = iinfo.max - iinfo.min
        self.omin       = - ran // 2
        self.omax       = ran // 2

        self.gulp       = gulp
        self.nchans     = nchans
        self.npol       = npol
        ## shapes
        self.gshape     = (self.gulp, self.nchans, self.npol)
        self.fshape     = (self.gulp, self.npol, self.nchans)
        ##
        # setup work arrays
        self.sdat       = np.zeros (self.gshape, dtype=np.float32)
        self.zerodat    = np.zeros (self.gshape, dtype=np.float32)
        self.dat        = np.zeros (self.fshape, dtype=self.odtype)

        ## setup output arrays
        self.dat_scl    = np.zeros ((self.npol, self.nchans), dtype=np.float32)
        self.dat_offs   = np.zeros ((self.npol, self.nchans), dtype=np.float32)
        self.means      = np.zeros ((1, self.nchans, self.npol), dtype=np.float32)
        self.scales     = np.zeros ((1, self.nchans, self.npol), dtype=np.float32)

    def _reset (self):
        """resets the arrays"""
        self.dat_scl[:]   = 0.
        self.dat_offs[:]  = 0.
        self.sdat[:]      = 0.
        self.dat[:]       = 0


    def __call__ (self, gfb):
        """ call 
            gfb should be int16
        """
        if gfb.dtype != np.int16:
            raise ValueError ("should give int16")
        ## assume gfb.shape == self.gshape
        self._reset ()

        ##
        ## stokes computation
        self.sdat[...,0] = gfb[...,0] + gfb[...,2]
        self.sdat[...,1] = gfb[...,1]
        self.sdat[...,2] = gfb[...,3]
        self.sdat[...,3] = gfb[...,0] - gfb[...,2]

        ##
        ## offset computation
        self.means[0]    = self.sdat.mean (0)
        ### subtract means
        self.sdat        -= self.means
        ### copy into DAT_OFFS
        self.dat_offs[0] = self.means[0, :, 0]
        self.dat_offs[1] = self.means[0, :, 1]
        self.dat_offs[2] = self.means[0, :, 2]
        self.dat_offs[3] = self.means[0, :, 3]
        ##
        ## scale computation
        self.scales[0]   = self.sdat.std (0)
        ### safe divide scale
        # self.sdat        /= self.scales
        self.sdat[:]     = np.divide (self.sdat, self.scales, out=self.zerodat, where=self.scales!=0.)
        ### copy into DAT_SCL
        self.dat_scl[0]  = self.scales[0, :, 0]
        self.dat_scl[1]  = self.scales[0, :, 1]
        self.dat_scl[2]  = self.scales[0, :, 2]
        self.dat_scl[3]  = self.scales[0, :, 3]

        ##
        ## digitize
        #### sdat is (nsamps, nchans, npol)
        #### bdat is (nsamps, npol, nchans)
        self.dat[:,0,:] = np.clip (self.sdat[...,0], self.omin, self.omax)+self.omin
        self.dat[:,1,:] = np.clip (self.sdat[...,1], self.omin, self.omax)+self.omin
        self.dat[:,2,:] = np.clip (self.sdat[...,2], self.omin, self.omax)+self.omin
        self.dat[:,3,:] = np.clip (self.sdat[...,3], self.omin, self.omax)+self.omin


if __name__ == "__main__":
    print ()
    import matplotlib.pyplot as plt
    f = "/home/shining/work/ofrb/gmrt_R67/ddtc176_snippets/59348.2349622268_dm411.000_sn33.71_lof550.npz"
    ff = np.load (f)
    keys = list(ff.keys())
    print ("Keys = ", keys)
    dd = ff['dd']
    print (dd.shape)
    ##
    rdi  = Redigitize (*dd.shape, odtype=np.uint16)
    rdi (np.int16(dd))
    ##






