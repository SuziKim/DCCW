import enum

class GraphCutDirection(enum.Enum):
	Forward = 0
	Reverse = 1
	
class InitialSeedSelectionMode(enum.Enum):
	CostFunction = 0
	TSPGraphCut = 1

class InitialSeedCostMode(enum.Enum):
	WoLightness = 0
	WoIsolation = 1
	All = 3


class IsolationMode(enum.Enum):
	ClosestDistanceOnly = 0
	# WeightedRank = 1
	# Average = 2 # For experiments, average distance between every points
	LoOP = 3 # Local Outlier Probabilities
	'''
	ClosestDistanceOnly:
		Consider the single closest distance 

	WeightedRank: 

	'''

class NormalizationMode(enum.Enum):
	NoNormalization = 5
	MinMax = 0
	Mean = 1
	Zscore = 2
	MinMaxAfterMean = 3
	MinMaxAfterZscore = 4

class SimMeasureStrategy(enum.Enum):
	DynamicClosestColorWarping = 0
	DynamicClosestColorWarpingConnected = 1

	ClassicHausdorffDistance = 2
	'''
	C-HD = max(h(A,B), h(B,A))
	h(A,B) = max_{a in A} { min_{b in B} { d(a,b) } }
	'''

	ModifiedHausdorffDistance = 3 # dubussion and jain
	'''
	M-HD = max(h'(A,B) h'(B,A))
	h'(A,B) = 1/N_{A} * sum_{a in A} { min_{b in B} { d(a,b) } }
	'''
	
	LeastTrimmedSquareHausdorffDistance = 4
	'''
	LTS-HD (Sim et al. 1999)
	LST-HD = max(h_{LTS}(A,B), h_{LTS}(B,A))
	h_{LTS} = 1/H * sum_{i=1..H} { d_{B}(a)_{i} }
	H = h * N_{A}, 0 <= h <= 1
	d_{B}(a)_{i}: i-th distance value in the sorted sequence d_{B}(x)_{1} <= d_{B}(x)_{2} <= ... <= d_{B}(x)_{N_{A}} 

	It calculates the distance between a and top h points only.
	'''

	PairwiseAverage = 5

	SignatureQuadraticFormDistance = 6
	'''
	SQFD (Beecks et al. 2010)
	Palette A (length k), Palette B (lengh m)
	SQFD = sqrt ( sum_{i=1..k} { sum_{j=1..k} { f_{s}(C^A_i, C^A_j) } } \
			+ sum_{i=1..m} { sum_{j=1..m} { f_{s}(C^B_i, C^B_j) } } \
			- 2 * sum_{i=1..k} { sum_{j=1..m} { f_{s}(C^A_i, C^B_j) } } )
	f_{s}(x,y) = 1 / (1 + L2(x,y)) when L2 is an Euclidean distance between x, y.
	We use CIDE200 distance instead of L2.

	SQFD represents the sum of
		1) + color differences of colors in palette A
		2) + color differences of colors in palette B
		3) - color differences of colors between palette A and palette B
	
	So, if we query palette A, 1) is fixed value for all similarities
	2), 3) differs. but, 3) is related to pairwise distance, and 2) is unnecessary.
	'''

	MinimumColorDifference = 7
	'''
	MCD (Pan and Westland 2018)
	MCD = ( m(A,B) + m(B,A) ) / 2
	m(A,B) = 1/N_{A} * sum_{a in A} { min_{b in B} { d(a,b) } }
	d(a,b) is CIDE2000 distance between color a and b.
	'''

	MergedPaletteHistogramSimilarityMeasure = 8
	'''
	MPHSM (Po and Wong 2004) w/ Td=15
	1) Generate Common Palettes (size: N_{m})
	2) Refine Pa and Pb with common palettes
	3) MPHSM(A,B) = sum_{i=1 to N_{m}} { min( p_{1mi}, p_{2mi} ) }
	'''

	ColorBasedEarthMoversDistance = 9
	'''
	CEMD (Skaff et al. 2011)
	EMD with CIED2000 Distance w/threshold = 20 (as paper mentioned)

	The library requires a citation: https://github.com/wmayner/pyemd
	'''

	MinimumBipartiteMatchingError = 10
	'''
	MBME (Lin and Hanrahan 2013)

	We define the distance between two themes to be the minimum total error from a bipartite matching of each color in one theme to a color in the other theme.
	'''

	DynamicTimeWarping = 11

	# SumOfDTWAndDCCW = 15

class TSPSolverMode(enum.Enum):
	LKH = 0 # elkai, http://akira.ruc.dk/~keld/
	NN = 1
	GA = 2
	FIA = 3 # used in Spira and Malah 2001
	SA = 4
	ACO = 6
	ACO50 = 7
	ACO10 = 8
	ACO5 = 9
	ACO2 = 10

class LabDistanceMode(enum.Enum):
	Euclidean = 0
	CIEDE2000 = 1

class SinglePaletteSortMode(enum.Enum):
	Luminance = {'id': 1, 'tsp_solver': None, 'lab_distance': None}
	HSV = {'id': 2, 'tsp_solver': None, 'lab_distance': None}
	NN_Euclidean = {'id': 3, 'tsp_solver': TSPSolverMode.NN, 'lab_distance': LabDistanceMode.Euclidean}
	NN_CIEDE2000 = {'id': 4, 'tsp_solver': TSPSolverMode.NN, 'lab_distance': LabDistanceMode.CIEDE2000}
	GA_Euclidean = {'id': 5, 'tsp_solver': TSPSolverMode.GA, 'lab_distance': LabDistanceMode.Euclidean}
	GA_CIEDE2000 = {'id': 6, 'tsp_solver': TSPSolverMode.GA, 'lab_distance': LabDistanceMode.CIEDE2000}
	FIA_Euclidean = {'id': 7, 'tsp_solver': TSPSolverMode.FIA, 'lab_distance': LabDistanceMode.Euclidean}
	FIA_CIEDE2000 = {'id': 8, 'tsp_solver': TSPSolverMode.FIA, 'lab_distance': LabDistanceMode.CIEDE2000}
	SA_Euclidean = {'id': 9, 'tsp_solver': TSPSolverMode.SA, 'lab_distance': LabDistanceMode.Euclidean}
	SA_CIEDE2000 = {'id': 10, 'tsp_solver': TSPSolverMode.SA, 'lab_distance': LabDistanceMode.CIEDE2000}
	ACO_Euclidean = {'id': 11, 'tsp_solver': TSPSolverMode.ACO, 'lab_distance': LabDistanceMode.Euclidean}
	ACO50_CIEDE2000 = {'id': 12, 'tsp_solver': TSPSolverMode.ACO50, 'lab_distance': LabDistanceMode.CIEDE2000}
	ACO10_CIEDE2000 = {'id': 15, 'tsp_solver': TSPSolverMode.ACO10, 'lab_distance': LabDistanceMode.CIEDE2000}
	ACO5_CIEDE2000 = {'id': 16, 'tsp_solver': TSPSolverMode.ACO5, 'lab_distance': LabDistanceMode.CIEDE2000}
	ACO2_CIEDE2000 = {'id': 17, 'tsp_solver': TSPSolverMode.ACO2, 'lab_distance': LabDistanceMode.CIEDE2000}
	LKH_Euclidean = {'id': 13, 'tsp_solver': TSPSolverMode.LKH, 'lab_distance': LabDistanceMode.Euclidean}
	LKH_CIEDE2000 = {'id': 14, 'tsp_solver': TSPSolverMode.LKH, 'lab_distance': LabDistanceMode.CIEDE2000}


class MultiplePalettesSortMode(enum.Enum):
	Merge_LKH = 0 
	Separate_Luminance = 1
	Separate_HSV = 2
	Separate_LKH = 3
	Merge_Luminance = 4
	Merge_HSV = 5
	BPS = 6
	Improved_BPS = 7


class MergeCutType(enum.Enum):
	Without_Cutting = 0
	With_Cutting = 1