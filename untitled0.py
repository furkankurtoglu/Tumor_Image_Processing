# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:39:26 2019

@author: Furkan
"""

import numpy as np
from nibabel import trackvis as tv
from dipy.data import get_fnames
from dipy.viz import window, actor

fname = get_fnames('fornix')
streams, hdr = tv.read(fname)
streamlines = [i[0] for i in streams]