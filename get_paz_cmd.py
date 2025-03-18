# coding: utf-8
import os
import subprocess as sp
import numpy as np

CMD="psrstat -Q -jT -c int:wt {ar}"

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



    ##
######################################################
if __name__ == "__main__":
    args = get_args ()
    for a in args.ar:
        action ( a, og=args.og )
######################################################


