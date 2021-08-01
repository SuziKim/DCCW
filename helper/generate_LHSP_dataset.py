import sys
sys.path.append(".")
sys.path.append("..")

from experiments.similarity_dataset_processor import *
from experiments.experiments_enums import *

target_datasets = [LHSPType.LHSP_k5_jitter0_replacement0, LHSPType.LHSP_k5_jitter5_replacement0, LHSPType.LHSP_k5_jitter10_replacement0, LHSPType.LHSP_k5_jitter15_replacement0, LHSPType.LHSP_k5_jitter0_replacement1, LHSPType.LHSP_k5_jitter5_replacement1, LHSPType.LHSP_k5_jitter10_replacement1, LHSPType.LHSP_k5_jitter15_replacement1, LHSPType.LHSP_k10_jitter0_replacement0, LHSPType.LHSP_k10_jitter5_replacement0, LHSPType.LHSP_k10_jitter10_replacement0, LHSPType.LHSP_k10_jitter15_replacement0, LHSPType.LHSP_k10_jitter0_replacement1, LHSPType.LHSP_k10_jitter5_replacement1, LHSPType.LHSP_k10_jitter10_replacement1, LHSPType.LHSP_k10_jitter15_replacement1, LHSPType.LHSP_k15_jitter0_replacement0, LHSPType.LHSP_k15_jitter5_replacement0, LHSPType.LHSP_k15_jitter10_replacement0, LHSPType.LHSP_k15_jitter15_replacement0, LHSPType.LHSP_k15_jitter0_replacement2, LHSPType.LHSP_k15_jitter5_replacement2, LHSPType.LHSP_k15_jitter10_replacement2, LHSPType.LHSP_k15_jitter15_replacement2, LHSPType.LHSP_k20_jitter0_replacement0, LHSPType.LHSP_k20_jitter5_replacement0, LHSPType.LHSP_k20_jitter10_replacement0, LHSPType.LHSP_k20_jitter15_replacement0, LHSPType.LHSP_k20_jitter0_replacement2, LHSPType.LHSP_k20_jitter5_replacement2, LHSPType.LHSP_k20_jitter10_replacement2, LHSPType.LHSP_k20_jitter15_replacement2, LHSPType.LHSP_k25_jitter0_replacement0, LHSPType.LHSP_k25_jitter5_replacement0, LHSPType.LHSP_k25_jitter10_replacement0, LHSPType.LHSP_k25_jitter15_replacement0, LHSPType.LHSP_k25_jitter0_replacement3, LHSPType.LHSP_k25_jitter5_replacement3, LHSPType.LHSP_k25_jitter10_replacement3, LHSPType.LHSP_k25_jitter15_replacement3, LHSPType.LHSP_k30_jitter0_replacement0, LHSPType.LHSP_k30_jitter5_replacement0, LHSPType.LHSP_k30_jitter10_replacement0, LHSPType.LHSP_k30_jitter15_replacement0, LHSPType.LHSP_k30_jitter0_replacement3, LHSPType.LHSP_k30_jitter5_replacement3, LHSPType.LHSP_k30_jitter10_replacement3, LHSPType.LHSP_k30_jitter15_replacement3]

for target_dataset in target_datasets:
	print('genearting %s' % target_dataset.name)
	dataset_processor = SimilarityDatasetProcessor(target_dataset, use_saved_database=True)
