# coding: utf-8
import os
import astropy.io.fits as aif

txt = """ChebyModelSet 1 segments
ChebyModel BEGIN
PSRNAME FRBR3
SITENAME GMRT
TIME_RANGE {imjd:d} {jmjd:d}
FREQ_RANGE 550 750
DISPERSION_CONSTANT 0.0
NCOEFF_TIME 2
NCOEFF_FREQ 2
COEFFS 1.0 0.0
COEFFS 1.0 0.0
ChebyModel END"""

def find_table ( names, tab ):
    for i,n in enumerate ( names ):
        if n == tab:
            return i
    raise RuntimeError (f"No {tab} found")

def get_args ():
    import argparse as agp
    ap   = agp.ArgumentParser ('pcol_t2pred', description='add period column and t2predictor', epilog='Part of GMRT/FRB')
    add  = ap.add_argument
    add ('ar', help='archive file', nargs='+')
    # add ('-v', '--verbose', help='Verbose', dest='v', action='store_true')
    return ap.parse_args ()

def action ( ar, oar ):
    """ input/output file """
    ##
    f = aif.open ( ar )
    fpri = f[0]

    imjd = fpri.header['STT_IMJD']

    iimjd = imjd - 1
    jjmjd = imjd + 1

    ## get sub
    SUBIDX = find_table ( [fi.name for fi in f], 'SUBINT' )
    PCOIDX = find_table ( [fi.name for fi in f], 'POLYCO' )
    fsub   = f[SUBIDX]
    fpco   = f[PCOIDX]

    f0     = fpco.data['REF_F0'][0]
    p0     = 1.0 / f0

    ## prepare fsub
    pcol   = aif.Column ( name='PERIOD', format='1D', unit='s', array=[[p0]] ) 

    # print (f" ar, oar, pcol,imjd,smjd={p0},{iimjd},{jjmjd}, {ar}, {oar}")

    ## prepare T2predictor table
    pxt = txt.format(imjd=iimjd, jmjd=jjmjd).split('\n')
    poc_columns = [ aif.Column(name="PREDICT", format="128A", array=pxt ) ]
    t2_hdr = aif.Header ()
    t2_hdu = aif.BinTableHDU( aif.FITS_rec.from_columns ( poc_columns ), name='t2predict', header=t2_hdr )

    ## write output
    ofits = aif.HDUList()
    ofits.append ( fpri )
    ofits.append ( fsub )
    ofits.append ( t2_hdu )
    ofits.writeto ( oar )

    ##
######################################################
if __name__ == "__main__":
    args = get_args ()
    for a in args.ar:
        p,_ = os.path.splitext ( a )
        action ( a, p+".arpt2" )
######################################################


