import os
import enum
import csv
import random

from dccw.single_palette_sorter import * 
from dccw.color_palettes import *
from dccw.color import *

# ==========================================
# SPS
# ==========================================
allowed_SPS_modes = [SinglePaletteSortMode.LKH_CIEDE2000,
					SinglePaletteSortMode.Luminance,
					SinglePaletteSortMode.HSV,
					SinglePaletteSortMode.GA_CIEDE2000, 
					SinglePaletteSortMode.FIA_CIEDE2000, 
					SinglePaletteSortMode.SA_CIEDE2000, 
					SinglePaletteSortMode.ACO50_CIEDE2000,
					SinglePaletteSortMode.ACO10_CIEDE2000,
					SinglePaletteSortMode.ACO5_CIEDE2000,
					SinglePaletteSortMode.ACO2_CIEDE2000]


# ==========================================
# FM100P
# ==========================================
class FM100PType(enum.Enum):
	FM100P_10 = 10
	FM100P_15 = 15
	FM100P_20 = 20
	FM100P_25 = 25
	FM100P_30 = 30
	FM100P_35 = 35
	FM100P_40 = 40


# ==========================================
# KHTP
# ==========================================
class KHTPType(enum.Enum):
	KHTP_interpolation0_jitter0 = "KHTP-interpolation0-jitter0"
	KHTP_interpolation0_jitter10 = "KHTP-interpolation0-jitter10"
	KHTP_interpolation0_jitter15 = "KHTP-interpolation0-jitter15"
	KHTP_interpolation0_jitter5 = "KHTP-interpolation0-jitter5"
	KHTP_interpolation1_jitter0 = "KHTP-interpolation1-jitter0"
	KHTP_interpolation1_jitter10 = "KHTP-interpolation1-jitter10"
	KHTP_interpolation1_jitter15 = "KHTP-interpolation1-jitter15"
	KHTP_interpolation1_jitter5 = "KHTP-interpolation1-jitter5"
	KHTP_interpolation3_jitter0 = "KHTP-interpolation3-jitter0"
	KHTP_interpolation3_jitter10 = "KHTP-interpolation3-jitter10"
	KHTP_interpolation3_jitter15 = "KHTP-interpolation3-jitter15"
	KHTP_interpolation3_jitter5 = "KHTP-interpolation3-jitter5"
	KHTP_interpolation5_jitter0 = "KHTP-interpolation5-jitter0"
	KHTP_interpolation5_jitter10 = "KHTP-interpolation5-jitter10"
	KHTP_interpolation5_jitter15 = "KHTP-interpolation5-jitter15"
	KHTP_interpolation5_jitter5 = "KHTP-interpolation5-jitter5"

class KHTPJitterOffset(enum.Enum):
	jitter5 = 5
	jitter10 = 10
	jitter15 = 15

class KHTPInterpolationCount(enum.Enum):
	interpolation1 = 1
	interpolation3 = 3
	interpolation5 = 5

class AllowedKHTPCategory(enum.Enum):
	KHTP_O = [khtp_type for khtp_type in KHTPType if 'interpolation0' in khtp_type.name and 'jitter0' in khtp_type.name]
	KHTP_I = [khtp_type for khtp_type in KHTPType if 'interpolation0' not in khtp_type.name and 'jitter0' in khtp_type.name]
	KHTP_J = [khtp_type for khtp_type in KHTPType if 'interpolation0' in khtp_type.name and 'jitter0' not in khtp_type.name]
	KHTP_J_I = [khtp_type for khtp_type in KHTPType if 'interpolation0' not in khtp_type.name and 'jitter0' not in khtp_type.name]


# ==========================================
# LHSP
# ==========================================
class LHSPType(enum.Enum):
	LHSP_k5_jitter0_replacement0 = 0
	LHSP_k5_jitter5_replacement0 = 3
	LHSP_k5_jitter10_replacement0 = 4
	LHSP_k5_jitter15_replacement0 = 5
	LHSP_k5_jitter0_replacement1 = 1
	LHSP_k5_jitter5_replacement1 = 100
	LHSP_k5_jitter10_replacement1 = 101
	LHSP_k5_jitter15_replacement1 = 102

	LHSP_k10_jitter0_replacement0 = 20
	LHSP_k10_jitter5_replacement0 = 23
	LHSP_k10_jitter10_replacement0 = 24
	LHSP_k10_jitter15_replacement0 = 25
	LHSP_k10_jitter0_replacement1 = 103
	LHSP_k10_jitter5_replacement1 = 104
	LHSP_k10_jitter10_replacement1 = 105
	LHSP_k10_jitter15_replacement1 = 106

	LHSP_k15_jitter0_replacement0 = 30
	LHSP_k15_jitter5_replacement0 = 33
	LHSP_k15_jitter10_replacement0 = 34
	LHSP_k15_jitter15_replacement0 = 35
	LHSP_k15_jitter0_replacement2 = 107
	LHSP_k15_jitter5_replacement2 = 108
	LHSP_k15_jitter10_replacement2 = 109
	LHSP_k15_jitter15_replacement2 = 110

	LHSP_k20_jitter0_replacement0 = 10
	LHSP_k20_jitter5_replacement0 = 13
	LHSP_k20_jitter10_replacement0 = 14
	LHSP_k20_jitter15_replacement0 = 15
	LHSP_k20_jitter0_replacement2 = 111
	LHSP_k20_jitter5_replacement2 = 112
	LHSP_k20_jitter10_replacement2 = 113
	LHSP_k20_jitter15_replacement2 = 114

	LHSP_k25_jitter0_replacement0 = 40
	LHSP_k25_jitter5_replacement0 = 43
	LHSP_k25_jitter10_replacement0 = 44
	LHSP_k25_jitter15_replacement0 = 45
	LHSP_k25_jitter0_replacement3 = 115
	LHSP_k25_jitter5_replacement3 = 116
	LHSP_k25_jitter10_replacement3 = 117
	LHSP_k25_jitter15_replacement3 = 118

	LHSP_k30_jitter0_replacement0 = 50
	LHSP_k30_jitter5_replacement0 = 53
	LHSP_k30_jitter10_replacement0 = 54
	LHSP_k30_jitter15_replacement0 = 55
	LHSP_k30_jitter0_replacement3 = 119
	LHSP_k30_jitter5_replacement3 = 120
	LHSP_k30_jitter10_replacement3 = 121
	LHSP_k30_jitter15_replacement3 = 122

class LHSPPaletteType(enum.Enum):
	QueryPalettes = 0
	TargetPalettes = 1
	Swatches = 2

class AllowedLHSPCategory(enum.Enum):
	LHSP_O = [lhsp_type for lhsp_type in LHSPType if 'jitter0' in lhsp_type.name and 'replacement0' in lhsp_type.name]
	LHSP_R = [lhsp_type for lhsp_type in LHSPType if 'jitter0' in lhsp_type.name and 'replacement0' not in lhsp_type.name]
	LHSP_J = [lhsp_type for lhsp_type in LHSPType if 'jitter0' not in lhsp_type.name and 'replacement0' in lhsp_type.name]
	LHSP_J_R = [lhsp_type for lhsp_type in LHSPType if 'jitter0' not in lhsp_type.name and 'replacement0' not in lhsp_type.name]
	
	LHSP_5 = [lhsp_type for lhsp_type in LHSPType if 'k5' in lhsp_type.name]
	LHSP_10 = [lhsp_type for lhsp_type in LHSPType if 'k10' in lhsp_type.name]
	LHSP_15 = [lhsp_type for lhsp_type in LHSPType if 'k15' in lhsp_type.name]
	LHSP_20 = [lhsp_type for lhsp_type in LHSPType if 'k20' in lhsp_type.name]
	LHSP_25 = [lhsp_type for lhsp_type in LHSPType if 'k25' in lhsp_type.name]
	LHSP_30 = [lhsp_type for lhsp_type in LHSPType if 'k30' in lhsp_type.name]


# ==========================================
# Logging
# ==========================================
class ShowLogLevel(enum.Enum):
	ImportantOnly = 3
	IncludingMidLevel = 2


