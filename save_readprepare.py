"""
saves output from read_prepare
"""

import sys

import numpy as np

from read_prepare import read_prepare_2d


if __name__ == "__main__":
    ###
    arg = sys.argv[1]
    ###

    freqs, i, q, u, v, ei, eq, eu, ev = read_prepare_2d ( arg, 4, False )

    np.savez (arg+"_read2.npz", freqs=freqs, i=i, q=q, u=u, v=v, ei=ei, eq=eq, eu=eu, ev=ev)



