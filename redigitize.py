
import numpy as np


class Redigitize:
    """
    Class performing redigitization

    uGMRT filterbank is 16bit. 
    It carries the bandshape and all. 
    Need to whiten it

    Direct re-scaling into [0, ??] where ?? is <bitdepth>::max
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
        self.omin       = iinfo.min
        self.omax       = iinfo.max + 1
        self.oran       = self.omax - self.omin
        self.omid       = int (0.5 * (self.omin + self.omax))

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

        self.mins       = np.zeros ((1, self.nchans, self.npol), dtype=np.float32)
        self.maxs       = np.zeros ((1, self.nchans, self.npol), dtype=np.float32)
        self.scales     = np.zeros ((1, self.nchans, self.npol), dtype=np.float32)
        self.means      = np.zeros ((1, self.nchans, self.npol), dtype=np.float32)

    def _reset (self):
        """resets the arrays"""
        self.dat_scl[:]   = 0.
        self.dat_offs[:]  = 0.
        self.sdat[:]      = 0.
        self.dat[:]       = 0


    def __call__ (self, gfb):
        """ call 
            gfb should be int16

            means should be omid
            scls should encapsulate entire dynamic range
        """
        if gfb.dtype != np.int16:
            raise ValueError ("should give int16")
        ## assume gfb.shape == self.gshape
        self._reset ()

        ##
        ## stokes computation
        ## see https://arxiv.org/pdf/2004.08542.pdf
        ## XXX RR, RL*, LL, R*L
        #self.sdat[...,0] = gfb[...,0] + gfb[...,2]
        #self.sdat[...,1] = -gfb[...,1] - gfb[...,3]
        #self.sdat[...,2] = gfb[...,1] - gfb[...,3]
        #self.sdat[...,3] = gfb[...,2] - gfb[...,0]
        ## 2021-11-24 SB: see discussion in POL_TYPE in obsinfo
        ## XXX AABBCRCI === RR,LL,RL*,R*L
        self.sdat[...,0] = gfb[...,0]
        self.sdat[...,1] = gfb[...,2]
        self.sdat[...,2] = gfb[...,1]
        self.sdat[...,3] = gfb[...,3]
        ## POL_TYPE = "IVQU" circular basis
        #self.sdat[...,0] = gfb[...,0] + gfb[...,2]
        #self.sdat[...,1] = gfb[...,2] - gfb[...,0]
        #self.sdat[...,2] = -gfb[...,1] - gfb[...,3]
        #self.sdat[...,3] = gfb[...,1] - gfb[...,3]


        ##
        ## min, max
        self.mins[0]     = self.sdat.min (0)
        self.maxs[0]     = self.sdat.max (0)
        #### scales, means
        self.scales[0]   = (self.maxs - self.mins) / (self.omax - self.omin)
        self.means[0]    = self.mins - (self.scales * self.omin)
        ## get "X"
        ### subtract means
        self.sdat        -= self.means
        ### divide offsets
        self.sdat[:]     = np.divide (self.sdat, self.scales, out=self.zerodat, where=self.scales!=0.)
        ##
        ### copy into DAT_OFFS
        self.dat_offs[0] = self.means[0, :, 0]
        self.dat_offs[1] = self.means[0, :, 1]
        self.dat_offs[2] = self.means[0, :, 2]
        self.dat_offs[3] = self.means[0, :, 3]
        ### copy into DAT_SCL
        self.dat_scl[0]  = self.scales[0, :, 0]
        self.dat_scl[1]  = self.scales[0, :, 1]
        self.dat_scl[2]  = self.scales[0, :, 2]
        self.dat_scl[3]  = self.scales[0, :, 3]

        ##
        ## digitize
        # XXX 20211126 - SB: i am stupid enough to not add the scaling step here
        #### sdat is (nsamps, nchans, npol)
        #### bdat is (nsamps, npol, nchans)
        #self.dat[:,0,:] = np.clip (self.ofac * self.sdat[...,0], self.omin, self.omax)+self.omin
        # XXX SB: how dumb can i be? v this is obviously wrong
        #self.dat[:,0,:] = np.clip ( (self.ofac*self.sdat[...,0]) + self.omin, self.omin, self.omax)
        # XXX no digitization, capturing entire scale/offsets
        self.dat[:,0,:] = np.clip ( self.sdat[...,0], self.omin, self.omax)
        self.dat[:,1,:] = np.clip ( self.sdat[...,1], self.omin, self.omax)
        self.dat[:,2,:] = np.clip ( self.sdat[...,2], self.omin, self.omax)
        self.dat[:,3,:] = np.clip ( self.sdat[...,3], self.omin, self.omax)


if __name__ == "__main__":
    print ()
    dd   = np.random.randint (0, 32700, size=(1280, 64, 4), dtype=np.int16)
    ##
    rdi  = Redigitize (*dd.shape, odtype=np.uint8)
    rdi (dd)
    ##
    print (rdi.dat.mean(0))
    #print (rdi.dat.std(0))






