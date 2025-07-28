# coding: utf-8
import os
<<<<<<< HEAD
import subprocess as sp
import numpy as np

CMD="psrstat -Q -jT -c int:wt {ar}"
=======
import numpy as np
import astropy.io.fits as aif

def find_table ( names, tab ):
    for i,n in enumerate ( names ):
        if n == tab:
            return i
    raise RuntimeError (f"No {tab} found")

def get_args ():
    import argparse as agp
    ap   = agp.ArgumentParser ('get_paz_cmd', description='To get flagged channels', epilog='Part of GMRT/FRB')
    add  = ap.add_argument
    add ('ar', help='archive file', nargs='+')
    add('-t','--tag', help='other-settings', dest='og', nargs='+', type=str, default='')
    # add ('-v', '--verbose', help='Verbose', dest='v', action='store_true')
    return ap.parse_args ()

def action (ar, og):
    """ input file """
    _cmd = CMD.format(ar=ar)
    # print ( _cmd )
    runner = sp.check_output ( _cmd, shell=True,)
    txt  = runner.split()

    zz = np.where(np.fromstring(txt[1],dtype=int,sep=',')==0)[0]
    # os.system( 'paz ' + f + ' -Z ' + ','.join(map(str, zz)))
    # CMD    = "paz -z \"" + kill_list + f"\" {tag}.PR -e PRz"
    ZZZ = "paz {og} -z ".format(og=" ".join(og)) + ",".join(map(str,zz)) + " " + ar
    print ( ZZZ )

    # add ('-v', '--verbose', help='Verbose', dest='v', action='store_true')
    return ap.parse_args ()

def action ( ar):
    """ input file """
    ##
    bn = os.path.basename ( ar )
    tag = bn [ :bn.find('_R3') ] + "_R3"

    ##
    f = aif.open ( ar )
    fpri = f[0]


    ## get sub
    SUBIDX = find_table ( [fi.name for fi in f], 'SUBINT' )
    fsub   = f[SUBIDX]

    ## get DAT_WTS
    dw     = fsub.data['DAT_WTS'][0]
    ww     = np.where ( dw == 0. )[0]

    kill_list = ""
    for ik in ww:
        kill_list += f"{ik:d},"
    kill_list = kill_list[:-1]

    CMD    = "paz -z \"" + kill_list + f"\" {tag}.PR -e PRz"

    print (CMD)


    ##
######################################################
if __name__ == "__main__":
    args = get_args ()
    for a in args.ar:
        action ( a, og=args.og )
        action ( a )
######################################################


