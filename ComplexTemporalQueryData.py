"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2022/3/2
@description: null
"""
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Set, Union

from toolbox.data.DataSchema import DatasetCachePath
from toolbox.data.DatasetSchema import BaseDatasetSchema
from toolbox.data.functional import read_cache, cache_data


class ICEWS14(BaseDatasetSchema):
    def __init__(self, home: str = "data"):
        super(ICEWS14, self).__init__("ICEWS14", home)


class ICEWS05_15(BaseDatasetSchema):
    def __init__(self, home: str = "data"):
        super(ICEWS05_15, self).__init__("ICEWS05-15", home)

class ComplexQueryDatasetCachePath(DatasetCachePath):
    def __init__(self, cache_path: Path):
        DatasetCachePath.__init__(self, cache_path)
        self.train_queries_answers_path = self.cache_path / "train-queries-answers.pkl"
        self.train_queries_path = self.cache_path / "train-queries.pkl"
        self.train_answers_path = self.cache_path / "train-answers.pkl"
        self.valid_queries_path = self.cache_path / "valid-queries.pkl"
        self.valid_hard_answers_path = self.cache_path / "valid-hard-answers.pkl"
        self.valid_easy_answers_path = self.cache_path / "valid-easy-answers.pkl"
        self.test_queries_path = self.cache_path / "test-queries.pkl"
        self.test_hard_answers_path = self.cache_path / "test-hard-answers.pkl"
        self.test_easy_answers_path = self.cache_path / "test-easy-answers.pkl"
        self.stats_path = self.cache_path / "stats.txt"

QueryStructure = str
class ComplexQueryData:
    def __init__(self, cache_path: ComplexQueryDatasetCachePath):
        self.cache_path = cache_path
        self.train_queries_answers: Dict[QueryStructure, List[Tuple[QueryFlattenIds, Set[int]]]] = {}
        self.train_queries: Dict[QueryStructure, Set[QueryFlattenIds]] = {}
        self.train_answers: Dict[QueryFlattenIds, Set[int]] = {}
        self.valid_queries: Dict[QueryStructure, Set[QueryFlattenIds]] = {}
        self.valid_hard_answers: Dict[QueryFlattenIds, Set[int]] = {}
        self.valid_easy_answers: Dict[QueryFlattenIds, Set[int]] = {}
        self.test_queries: Dict[QueryStructure, Set[QueryFlattenIds]] = {}
        self.test_hard_answers: Dict[QueryFlattenIds, Set[int]] = {}
        self.test_easy_answers: Dict[QueryFlattenIds, Set[int]] = {}
        self.nentity: int = 0
        self.nrelation: int = 0