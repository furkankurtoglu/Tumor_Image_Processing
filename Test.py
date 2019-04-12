# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:57:05 2019

@author: fkurtog
"""

from dipy.data import get_fnames
import matplotlib.pyplot as plt
from nibabel import trackvis as tv

fname = get_fnames('fornix')

streams, hdr = tv.read(fname)

streamlines = [i[0] for i in streams]

plt.imshow(streamlines)