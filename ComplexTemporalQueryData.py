"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2022/3/2
@description: null
"""
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Set, Union, Any

import expression
from expression.ParamSchema import placeholder2sample, get_param_name_list, get_placeholder_list
from expression.TFLEX_DSL import query_structures
from toolbox.data.DataSchema import DatasetCachePath, BaseData
from toolbox.data.DatasetSchema import RelationalTripletDatasetSchema
from toolbox.data.functional import read_cache, cache_data
from toolbox.utils.Progbar import Progbar


class ICEWS14(RelationalTripletDatasetSchema):
    def __init__(self, home: Union[Path, str] = "data"):
        super(ICEWS14, self).__init__("ICEWS14", home)

    def get_data_paths(self) -> Dict[str, Path]:
        return {
            'train': self.get_dataset_path_child('train'),
            'test': self.get_dataset_path_child('test'),
            'valid': self.get_dataset_path_child('valid'),
        }

    def get_dataset_path(self):
        return self.root_path


class ICEWS05_15(RelationalTripletDatasetSchema):
    def __init__(self, home: Union[Path, str] = "data"):
        super(ICEWS05_15, self).__init__("ICEWS05-15", home)

    def get_data_paths(self) -> Dict[str, Path]:
        return {
            'train': self.get_dataset_path_child('train'),
            'test': self.get_dataset_path_child('test'),
            'valid': self.get_dataset_path_child('valid'),
        }

    def get_dataset_path(self):
        return self.root_path


class TemporalKnowledgeDatasetCachePath(DatasetCachePath):
    def __init__(self, cache_path: Path):
        DatasetCachePath.__init__(self, cache_path)
        self.cache_all_triples_path = self.cache_path / 'triplets_all.pkl'
        self.cache_train_triples_path = self.cache_path / 'triplets_train.pkl'
        self.cache_test_triples_path = self.cache_path / 'triplets_test.pkl'
        self.cache_valid_triples_path = self.cache_path / 'triplets_valid.pkl'

        self.cache_all_triples_ids_path = self.cache_path / 'triplets_ids_all.pkl'
        self.cache_train_triples_ids_path = self.cache_path / 'triplets_ids_train.pkl'
        self.cache_test_triples_ids_path = self.cache_path / 'triplets_ids_test.pkl'
        self.cache_valid_triples_ids_path = self.cache_path / 'triplets_ids_valid.pkl'

        self.cache_all_entities_path = self.cache_path / 'entities.pkl'
        self.cache_all_relations_path = self.cache_path / 'relations.pkl'
        self.cache_all_timestamps_path = self.cache_path / 'timestamps.pkl'
        self.cache_entities_ids_path = self.cache_path / "entities_ids.pkl"
        self.cache_relations_ids_path = self.cache_path / "relations_ids.pkl"
        self.cache_timestamps_ids_path = self.cache_path / "timestamps_ids.pkl"

        self.cache_idx2entity_path = self.cache_path / 'idx2entity.pkl'
        self.cache_idx2relation_path = self.cache_path / 'idx2relation.pkl'
        self.cache_idx2timestamp_path = self.cache_path / 'idx2timestamp.pkl'
        self.cache_entity2idx_path = self.cache_path / 'entity2idx.pkl'
        self.cache_relation2idx_path = self.cache_path / 'relation2idx.pkl'
        self.cache_timestamps2idx_path = self.cache_path / 'timestamp2idx.pkl'

        # self.cache_sro_t_path = self.cache_path / 'sro_t.pkl'
        # self.cache_sro_t_train_path = self.cache_path / 'sro_t_train.pkl'
        # self.cache_sro_t_valid_path = self.cache_path / 'sro_t_valid.pkl'
        # self.cache_sro_t_test_path = self.cache_path / 'sro_t_test.pkl'
        #
        # self.cache_srt_o_path = self.cache_path / 'srt_o.pkl'
        # self.cache_srt_o_train_path = self.cache_path / 'srt_o_train.pkl'
        # self.cache_srt_o_valid_path = self.cache_path / 'srt_o_valid.pkl'
        # self.cache_srt_o_test_path = self.cache_path / 'srt_o_test.pkl'


def read_triple_srot(file_path: Union[str, Path]) -> List[Tuple[str, str, str, str]]:
    """
    return [(lhs, rel, rhs, timestamp)]
              s    r    o       t
    """
    with open(str(file_path), 'r', encoding='utf-8') as fr:
        triple = set()
        for line in fr.readlines():
            lhs, rel, rhs, timestamp = line.strip().split('\t')
            triple.add((lhs, rel, rhs, timestamp))
    return list(triple)


TYPE_MAPPING_sro_t = Dict[int, Dict[int, Dict[int, Set[int]]]]
TYPE_MAPPING_srt_o = Dict[int, Dict[int, Dict[int, Set[int]]]]


def build_map_sro2t_and_srt2o(triplets: List[Tuple[int, int, int, int]]) -> Tuple[TYPE_MAPPING_sro_t, TYPE_MAPPING_srt_o]:
    """ Function to read the list of tails for the given head and relation pair. """
    sro_t = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    srt_o = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for s, r, o, t in triplets:
        sro_t[s][r][o].add(t)
        srt_o[s][r][t].add(o)
    return sro_t, srt_o


class TemporalKnowledgeData(BaseData):
    """ The class is the main module that handles the knowledge graph.

        KnowledgeGraph is responsible for downloading, parsing, processing and preparing
        the training, testing and validation dataset.

        Args:
            dataset (RelationalTripletDatasetSchema): custom dataset.
            cache_path (TemporalKnowledgeDatasetCachePath): cache path.

        Attributes:
            dataset (RelationalTripletDatasetSchema): custom dataset.
            cache_path (TemporalKnowledgeDatasetCachePath): cache path.

            all_relations (list):list of all the relations.
            all_entities (list): List of all the entities.
            all_timestamps (list): List of all the timestamps.

            entity2idx (dict): Dictionary for mapping string name of entities to unique numerical id.
            idx2entity (dict): Dictionary for mapping the entity id to string.
            relation2idx (dict): Dictionary for mapping string name of relations to unique numerical id.
            idx2relation (dict): Dictionary for mapping the relation id to string.
            timestamp2idx (dict): Dictionary for mapping string name of timestamps to unique numerical id.
            idx2timestamp (dict): Dictionary for mapping the timestamp id to string.

        Examples:
            >>> from ComplexTemporalQueryData import ICEWS14, TemporalKnowledgeDatasetCachePath, TemporalKnowledgeData
            >>> dataset = ICEWS14()
            >>> cache = TemporalKnowledgeDatasetCachePath(dataset.cache_path)
            >>> data = TemporalKnowledgeData(dataset=dataset, cache_path=cache)
            >>> data.preprocess_data_if_needed()

    """

    def __init__(self,
                 dataset: RelationalTripletDatasetSchema,
                 cache_path: TemporalKnowledgeDatasetCachePath):
        BaseData.__init__(self, dataset, cache_path)
        self.dataset = dataset
        self.cache_path = cache_path

        # KG data structure stored in triplet format
        self.all_triples: List[Tuple[str, str, str, str]] = []
        self.train_triples: List[Tuple[str, str, str, str]] = []
        self.test_triples: List[Tuple[str, str, str, str]] = []
        self.valid_triples: List[Tuple[str, str, str, str]] = []

        self.all_triples_ids: List[Tuple[int, int, int, int]] = []
        self.train_triples_ids: List[Tuple[int, int, int, int]] = []
        self.test_triples_ids: List[Tuple[int, int, int, int]] = []
        self.valid_triples_ids: List[Tuple[int, int, int, int]] = []

        self.all_relations: List[str] = []
        self.all_entities: List[str] = []
        self.all_timestamps: List[str] = []
        self.entities_ids: List[int] = []
        self.relations_ids: List[int] = []
        self.timestamps_ids: List[int] = []

        self.entity2idx: Dict[str, int] = {}
        self.idx2entity: Dict[int, str] = {}
        self.relation2idx: Dict[str, int] = {}
        self.idx2relation: Dict[int, str] = {}
        self.timestamp2idx: Dict[str, int] = {}
        self.idx2timestamp: Dict[int, str] = {}

        self.sro_t: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
        self.sro_t_train: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
        self.sro_t_valid: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
        self.sro_t_test: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)

        self.srt_o: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
        self.srt_o_train: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
        self.srt_o_valid: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
        self.srt_o_test: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)

        # meta
        self.entity_count = 0
        self.relation_count = 0
        self.timestamp_count = 0
        self.valid_triples_count = 0
        self.test_triples_count = 0
        self.train_triples_count = 0
        self.triple_count = 0

    def read_all_origin_data(self):
        self.read_all_triplets()

    def read_all_triplets(self):
        self.train_triples = read_triple_srot(self.dataset.data_paths['train'])
        self.valid_triples = read_triple_srot(self.dataset.data_paths['valid'])
        self.test_triples = read_triple_srot(self.dataset.data_paths['test'])
        self.all_triples = self.train_triples + self.valid_triples + self.test_triples

        self.valid_triples_count = len(self.valid_triples)
        self.test_triples_count = len(self.test_triples)
        self.train_triples_count = len(self.train_triples)
        self.triple_count = self.valid_triples_count + self.test_triples_count + self.train_triples_count

    def transform_all_data(self):
        self.transform_entities_relations_timestamps()
        self.transform_mappings()
        self.transform_all_triplets_ids()

        self.transform_entity_ids()
        self.transform_relation_ids()
        self.transform_timestamp_ids()

        self.transform_mapping()

    def transform_entities_relations_timestamps(self):
        """ Function to read the entities. """
        entities: Set[str] = set()
        relations: Set[str] = set()
        timestamps: Set[str] = set()
        # print("entities_relations")
        # bar = Progbar(len(self.all_triples))
        # i = 0
        for s, r, o, t in self.all_triples:
            entities.add(s)
            relations.add(r)
            entities.add(o)
            timestamps.add(t)
            # i += 1
            # bar.update(i, [("h", h.split("/")[-1]), ("r", r.split("/")[-1]), ("t", t.split("/")[-1])])

        self.all_entities = sorted(list(entities))
        self.all_relations = sorted(list(relations))
        self.all_timestamps = sorted(list(timestamps))

        self.entity_count = len(self.all_entities)
        self.relation_count = len(self.all_relations)
        self.timestamp_count = len(self.all_timestamps)

    def transform_mappings(self):
        """ Function to generate the mapping from string name to integer ids. """
        self.entity2idx = {v: k for k, v in enumerate(self.all_entities)}
        self.idx2entity = {v: k for k, v in self.entity2idx.items()}
        self.relation2idx = {v: k for k, v in enumerate(self.all_relations)}
        self.idx2relation = {v: k for k, v in self.relation2idx.items()}
        self.timestamp2idx = {v: k for k, v in enumerate(self.all_timestamps)}
        self.idx2timestamp = {v: k for k, v in self.timestamp2idx.items()}

    def transform_all_triplets_ids(self):
        entity2idx = self.entity2idx
        relation2idx = self.relation2idx
        timestamp2idx = self.timestamp2idx
        self.train_triples_ids = [(entity2idx[s], relation2idx[r], entity2idx[o], timestamp2idx[t]) for s, r, o, t in self.train_triples]
        self.test_triples_ids = [(entity2idx[s], relation2idx[r], entity2idx[o], timestamp2idx[t]) for s, r, o, t in self.test_triples]
        self.valid_triples_ids = [(entity2idx[s], relation2idx[r], entity2idx[o], timestamp2idx[t]) for s, r, o, t in self.valid_triples]
        self.all_triples_ids = self.train_triples_ids + self.valid_triples_ids + self.test_triples_ids

    def transform_entity_ids(self):
        entity2idx = self.entity2idx
        print("entities_ids")
        # bar = Progbar(len(self.all_entities))
        # i = 0
        for e in self.all_entities:
            self.entities_ids.append(entity2idx[e])
            # i += 1
            # bar.update(i, [("entity", e.split("/")[-1])])

    def transform_relation_ids(self):

        relation2idx = self.relation2idx

        print("relations_ids")
        # bar = Progbar(len(self.all_relations))
        # i = 0
        for r in self.all_relations:
            self.relations_ids.append(relation2idx[r])
            # i += 1
            # bar.update(i, [("relation", r.split("/")[-1])])

    def transform_timestamp_ids(self):
        timestamp2idx = self.timestamp2idx
        print("timestamps_ids")
        # bar = Progbar(len(self.all_relations))
        # i = 0
        for t in self.all_timestamps:
            self.timestamps_ids.append(timestamp2idx[t])
            # i += 1
            # bar.update(i, [("relation", r.split("/")[-1])])

    def transform_mapping(self):
        """ Function to read the list of tails for the given head and relation pair. """
        self.sro_t, self.srt_o = build_map_sro2t_and_srt2o(self.all_triples_ids)
        self.sro_t_train, self.srt_o_train = build_map_sro2t_and_srt2o(self.train_triples_ids)
        self.sro_t_valid, self.srt_o_valid = build_map_sro2t_and_srt2o(self.valid_triples_ids)
        self.sro_t_test, self.srt_o_test = build_map_sro2t_and_srt2o(self.test_triples_ids)

    def cache_all_data(self):
        """Function to cache the prepared dataset in the memory"""
        cache_data(self.all_triples, self.cache_path.cache_all_triples_path)
        cache_data(self.train_triples, self.cache_path.cache_train_triples_path)
        cache_data(self.test_triples, self.cache_path.cache_test_triples_path)
        cache_data(self.valid_triples, self.cache_path.cache_valid_triples_path)

        cache_data(self.all_triples_ids, self.cache_path.cache_all_triples_ids_path)
        cache_data(self.train_triples_ids, self.cache_path.cache_train_triples_ids_path)
        cache_data(self.test_triples_ids, self.cache_path.cache_test_triples_ids_path)
        cache_data(self.valid_triples_ids, self.cache_path.cache_valid_triples_ids_path)

        cache_data(self.all_entities, self.cache_path.cache_all_entities_path)
        cache_data(self.all_relations, self.cache_path.cache_all_relations_path)
        cache_data(self.all_timestamps, self.cache_path.cache_all_timestamps_path)
        cache_data(self.entities_ids, self.cache_path.cache_entities_ids_path)
        cache_data(self.relations_ids, self.cache_path.cache_relations_ids_path)
        cache_data(self.timestamps_ids, self.cache_path.cache_timestamps_ids_path)

        cache_data(self.idx2entity, self.cache_path.cache_idx2entity_path)
        cache_data(self.idx2relation, self.cache_path.cache_idx2relation_path)
        cache_data(self.idx2timestamp, self.cache_path.cache_idx2timestamp_path)
        cache_data(self.relation2idx, self.cache_path.cache_relation2idx_path)
        cache_data(self.entity2idx, self.cache_path.cache_entity2idx_path)
        cache_data(self.timestamp2idx, self.cache_path.cache_timestamps2idx_path)

        # pickle unable to save defaultdict
        # cache_data(self.sro_t, self.cache_path.cache_sro_t_path)
        # cache_data(self.srt_o, self.cache_path.cache_srt_o_path)
        # cache_data(self.sro_t_train, self.cache_path.cache_sro_t_train_path)
        # cache_data(self.srt_o_train, self.cache_path.cache_srt_o_train_path)
        # cache_data(self.sro_t_valid, self.cache_path.cache_sro_t_valid_path)
        # cache_data(self.srt_o_valid, self.cache_path.cache_srt_o_valid_path)
        # cache_data(self.sro_t_test, self.cache_path.cache_sro_t_test_path)
        # cache_data(self.srt_o_test, self.cache_path.cache_srt_o_test_path)

        cache_data(self.meta(), self.cache_path.cache_metadata_path)

    def load_cache(self, keys: List[str]):
        for key in keys:
            self.read_cache_data(key)

    def read_cache_data(self, key):
        """Function to read the cached dataset from the memory"""
        path = "cache_%s_path" % key
        if hasattr(self, key) and hasattr(self.cache_path, path):
            key_path = getattr(self.cache_path, path)
            value = read_cache(key_path)
            setattr(self, key, value)
            return value
        elif key == "meta":
            meta = read_cache(self.cache_path.cache_metadata_path)
            self.read_meta(meta)
        else:
            raise ValueError('Unknown cache data key %s' % key)

    def read_meta(self, meta):
        self.entity_count = meta["entity_count"]
        self.relation_count = meta["relation_count"]
        self.timestamp_count = meta["timestamp_count"]
        self.valid_triples_count = meta["valid_triples_count"]
        self.test_triples_count = meta["test_triples_count"]
        self.train_triples_count = meta["train_triples_count"]
        self.triple_count = meta["triple_count"]

    def meta(self) -> Dict[str, Any]:
        return {
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "timestamp_count": self.timestamp_count,
            "valid_triples_count": self.valid_triples_count,
            "test_triples_count": self.test_triples_count,
            "train_triples_count": self.train_triples_count,
            "triple_count": self.triple_count,
        }

    def dump(self) -> List[str]:
        """ Function to dump statistic information of a dataset """
        # dump key information
        dump = [
            "",
            "-" * 15 + "Metadata Info for Dataset: " + self.dataset.name + "-" * (15 - len(self.dataset.name)),
            "Total Training Triples   :%s" % self.train_triples_count,
            "Total Testing Triples    :%s" % self.test_triples_count,
            "Total validation Triples :%s" % self.valid_triples_count,
            "Total Entities           :%s" % self.entity_count,
            "Total Relations          :%s" % self.relation_count,
            "Total Timestamps         :%s" % self.timestamp_count,
            "-" * (30 + len("Metadata Info for Dataset: ")),
            "",
        ]
        return dump


"""
above is simple temporal kg
below is complex query data (logical reasoning) based on previous temporal kg
"""


class ComplexTemporalQueryDatasetCachePath(TemporalKnowledgeDatasetCachePath):
    def __init__(self, cache_path: Path):
        TemporalKnowledgeDatasetCachePath.__init__(self, cache_path)
        self.cache_train_queries_answers_path = self.cache_path / "train_queries_answers.pkl"
        self.cache_valid_queries_answers_path = self.cache_path / "valid_queries_answers.pkl"
        self.cache_test_queries_answers_path = self.cache_path / "test_queries_answers.pkl"


class ComplexQueryData(TemporalKnowledgeData):

    def __init__(self,
                 dataset: RelationalTripletDatasetSchema,
                 cache_path: ComplexTemporalQueryDatasetCachePath):
        TemporalKnowledgeData.__init__(self, dataset, cache_path)
        self.cache_path = cache_path
        # Dict[str, Dict[str, Union[List[str], List[Tuple[List[int], Set[int]]]]]]
        #       |                       |                     |          |
        #     structure name      args name list              |          |
        #                                    ids corresponding to args   |
        #                                                          answers id set
        # 1. `structure name` is the name of a function (named query function), parsed to AST and eval to get results.
        # 2. `args name list` is the arg list of query function.
        # 3. valid_queries_answers and test_queries_answers are the same type as train_queries_answers
        self.train_queries_answers: Dict[str, Dict[str, Union[List[str], List[Tuple[List[int], Set[int]]]]]] = {
            "Pe_aPt": {
                "args": ["e1", "r1", "e2", "r2", "e3"],
                "queries_answers": [
                    ([1, 2, 3, 4, 5], {2, 3, 5}),
                    ([1, 2, 3, 4, 5], {2, 3, 5}),
                    ([1, 2, 3, 4, 5], {2, 3, 5}),
                ]
            }
        }
        self.valid_queries_answers: Dict[str, Dict[str, Union[List[str], List[Tuple[List[int], Set[int]]]]]] = {}
        self.test_queries_answers: Dict[str, Dict[str, Union[List[str], List[Tuple[List[int], Set[int]]]]]] = {}
        # meta
        self.query_meta = {
            "Pe_aPt": {
                "queries_count": 1,
                "avg_answers_count": 1
            }
        }

    def transform_all_data(self):
        TemporalKnowledgeData.transform_all_data(self)
        # 1. add inverse relations
        max_relation_id = self.relation_count

        def append_reverse(triples, max_relation_id: int):
            res = []
            for s, r, o, t in triples:
                res.append((s, r, o, t))
                res.append((o, r + max_relation_id, s, t))
            return res

        all_triples_ids = append_reverse(self.all_triples_ids, max_relation_id)
        sro_t, srt_o = build_map_sro2t_and_srt2o(all_triples_ids)
        train_triples_ids = append_reverse(self.train_triples_ids, max_relation_id)
        valid_triples_ids = append_reverse(self.valid_triples_ids, max_relation_id)
        test_triples_ids = append_reverse(self.test_triples_ids, max_relation_id)
        relations_ids_with_reverse = self.relations_ids + [r + max_relation_id for r in self.relations_ids]
        train_sro_t, train_srt_o = build_map_sro2t_and_srt2o(train_triples_ids)
        valid_sro_t, valid_srt_o = build_map_sro2t_and_srt2o(train_triples_ids + valid_triples_ids)
        test_sro_t, test_srt_o = build_map_sro2t_and_srt2o(train_triples_ids + valid_triples_ids + test_triples_ids)

        # 1. 1-hop: Pe, Pt
        def build_one_hop(param_name_list: List[str], sro_t):
            queries_answers = []
            for s in sro_t:
                for r in sro_t[s]:
                    for o in sro_t[s][r]:
                        answers = sro_t[s][r][o]
                        if len(answers) > 0:
                            queries = [s, r, o]
                            queries_answers.append((queries, answers))
            return {
                "args": param_name_list,
                "queries_answers": queries_answers
            }

        self.train_queries_answers["Pe"] = build_one_hop(["e1", "r1", "t1"], train_srt_o)
        self.valid_queries_answers["Pe"] = build_one_hop(["e1", "r1", "t1"], valid_srt_o)
        self.test_queries_answers["Pe"] = build_one_hop(["e1", "r1", "t1"], test_srt_o)

        self.train_queries_answers["Pt"] = build_one_hop(["e1", "r1", "e2"], train_sro_t)
        self.valid_queries_answers["Pt"] = build_one_hop(["e1", "r1", "e2"], valid_sro_t)
        self.test_queries_answers["Pt"] = build_one_hop(["e1", "r1", "e2"], test_sro_t)

        # 2. multi-hop: Pe_aPt, Pe_bPt, etc
        # 2.1 parser
        train_parser = expression.SamplingParser(self.entities_ids, relations_ids_with_reverse, self.timestamps_ids, train_srt_o, train_sro_t)
        valid_parser = expression.SamplingParser(self.entities_ids, relations_ids_with_reverse, self.timestamps_ids, valid_srt_o, valid_sro_t)
        test_parser = expression.SamplingParser(self.entities_ids, relations_ids_with_reverse, self.timestamps_ids, test_srt_o, test_sro_t)

        # 2.2. sampling
        sample_count = self.triple_count // 2
        all_queries_answers = {}
        query_structure_name_list = query_structures.keys()

        def achieve_answers(train_query_structure_func, valid_query_structure_func, test_query_structure_func):
            answers = set()
            valid_answers = set()
            test_answers = set()
            placeholders = None
            while len(answers) <= 0:
                placeholders = get_placeholder_list(train_query_structure_func)
                sampling_query_answers = train_query_structure_func(*placeholders)
                if sampling_query_answers.answers is not None and len(sampling_query_answers.answers) > 0:
                    answers = sampling_query_answers.answers
                    valid_answers = valid_query_structure_func(*placeholders).answers
                    test_answers = test_query_structure_func(*placeholders).answers
                elif sampling_query_answers.timestamps is not None and len(sampling_query_answers.timestamps) > 0:
                    answers = sampling_query_answers.timestamps
                    valid_answers = valid_query_structure_func(*placeholders).timestamps
                    test_answers = test_query_structure_func(*placeholders).timestamps
                else:
                    answers = set()
                    valid_answers = set()
                    test_answers = set()
            queries = placeholder2sample(placeholders)
            return queries, answers, valid_answers, test_answers

        for query_structure_name in query_structure_name_list:
            train_query_structure_func = train_parser.eval(query_structure_name)
            valid_query_structure_func = valid_parser.eval(query_structure_name)
            test_query_structure_func = test_parser.eval(query_structure_name)
            train_queries_answers = []
            valid_queries_answers = []
            test_queries_answers = []
            bar = Progbar(sample_count)
            for i in range(sample_count):
                queries, answers, valid_answers, test_answers = achieve_answers(train_query_structure_func, valid_query_structure_func, test_query_structure_func)
                train_queries_answers.append((queries, answers))
                valid_queries_answers.append((queries, valid_answers))
                test_queries_answers.append((queries, test_answers))
                bar.update(i + 1, {"train": len(answers), "valid": len(valid_answers), "test": len(test_answers)})

            param_name_list = get_param_name_list(train_query_structure_func)
            self.train_queries_answers[query_structure_name] = {
                "args": param_name_list,
                "queries_answers": train_queries_answers
            }
            self.valid_queries_answers[query_structure_name] = {
                "args": param_name_list,
                "queries_answers": valid_queries_answers
            }
            self.test_queries_answers[query_structure_name] = {
                "args": param_name_list,
                "queries_answers": test_queries_answers
            }

        def avg_answers_count(qa):
            return sum([len(a) for q, a in qa]) / len(qa)

        def calculate_meta(query_name: str):
            train_qa = self.train_queries_answers[query_name]["queries_answers"]
            valid_qa = self.valid_queries_answers[query_name]["queries_answers"]
            test_qa = self.test_queries_answers[query_name]["queries_answers"]
            queries_answers = train_qa + valid_qa + test_qa
            self.query_meta[query_name] = {
                "queries_count": len(queries_answers),
                "avg_answers_count": avg_answers_count(queries_answers),
                "train": {
                    "queries_count": len(train_qa),
                    "avg_answers_count": avg_answers_count(train_qa),
                },
                "valid": {
                    "queries_count": len(valid_qa),
                    "avg_answers_count": avg_answers_count(valid_qa),
                },
                "test": {
                    "queries_count": len(test_qa),
                    "avg_answers_count": avg_answers_count(test_qa),
                },
            }
            print(query_name, self.query_meta[query_name])

        for k in self.test_queries_answers.keys():
            calculate_meta(k)

    def cache_all_data(self):
        TemporalKnowledgeData.cache_all_data(self)
        cache_data(self.train_queries_answers, self.cache_path.cache_train_queries_answers_path)
        cache_data(self.valid_queries_answers, self.cache_path.cache_valid_queries_answers_path)
        cache_data(self.test_queries_answers, self.cache_path.cache_test_queries_answers_path)

    def read_meta(self, meta):
        TemporalKnowledgeData.read_meta(self, meta)
        self.query_meta = meta["query_meta"]

    def meta(self) -> Dict[str, Any]:
        m = TemporalKnowledgeData.meta(self)
        m.update({
            "query_meta": self.query_meta,
        })
        return m

    def dump(self) -> List[str]:
        """ Function to dump statistic information of a dataset """
        # dump key information
        dump = TemporalKnowledgeData.dump(self)
        for k, v in self.query_meta.items():
            dump.insert(len(dump) - 3, f"{k} : {v}")
        return dump
