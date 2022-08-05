#!/usr/bin/env python
# coding: utf-8
import os,sys
import matplotlib
from PIL import Image

emissionlines = {
1033.82: "O VI",
1215.24: "Lyα",
1240.81: "N V",
1305.53: "O I",
1335.31: "C II",
1397.61: "Si IV",
1399.8: "Si IV + O IV",
1549.48: "C IV",
1640.4: "He II",
1665.85: "O III",
1857.4: "Al III",
1908.734: "C III",
2326.0: "C II",
2439.5: "Ne IV",
2799.117: "Mg II",
3346.79: "Ne V",
3426.85: "Ne VI",
3727.092: "", #"O II",
3729.875: "O II",
3889.0: "He I",
4072.3: "S II",
4102.89: "Hδ",
4341.68: "Hγ",
4364.436: "O III",
4862.68: "Hβ",
4932.603: "O III",
4960.295: "O III",
5008.240: "O III",
6302.046: "O I",
6365.536: "O I",
6529.03: "N I",
6549.86: "N II",
6564.61: "Hα",
6585.27: "N II",
6718.29: "S II",
6732.67: "S II",
}

absorptionlines = {
3934.777: "K",
3969.588: "H",
4305.61: "G",
5176.7: "Mg",
5895.6: "Na",
# 6496.9: "Ba II",
8500.36: "Ca II",
8544.44: "Ca II",
8664.52: "Ca II",
}

skylines = [5578.5, 5894.6, 6301.7, 7246.0]
lines = {**emissionlines, **absorptionlines}
