"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2022/2/21
@description: null
"""
import inspect
import random
from typing import List, Set, Dict, Union

from .ParamSchema import Placeholder, FixedQuery, get_placeholder_list
from .symbol import Interpreter

query_structures = {
    # plz read from right to left
    # 1. 1-hop
    # Pe and Pt, manually
    # 2. 2-hop
    "Pt_rPe": "def Pt_Pe(e1, r1, e2, r2, t1): return Pt(e1, r1, Pe(e2, r2, t1))", # r for right
    "Pt_lPe": "def Pt_Pe(e1, r1, t1, r2, e2): return Pt(Pe(e1, r1, t1), r2, e2)", # l for left
    "Pe_Pt": "def Pe_Pt(e1, r1, e2, r2, e3): return Pe(e1, r1, Pt(e2, r2, e3))",
    "Pe_aPt": "def Pe_aPt(e1, r1, e2, r2, e3): return Pe(e1, r1, after(Pt(e2, r2, e3)))",
    "Pe_bPt": "def Pe_bPt(e1, r1, e2, r2, e3): return Pe(e1, r1, before(Pt(e2, r2, e3)))",
    "Pe_nPt": "def Pe_bPt(e1, r1, e2, r2, e3): return Pe(e1, r1, next(Pt(e2, r2, e3)))",
    # 3. and
    "Pe_bPt": "def Pe_bPt(e1, r1, e2, r2, e3): return Pe(e1, r1, before(Pt(e2, r2, e3)))",
}


class BasicParser(Interpreter):
    """
    abstract class
    """

    def __init__(self, variables, neural_ops):
        alias = {
            "Pe": neural_ops["EntityProjection"],
            "Pt": neural_ops["TimeProjection"],
            "before": neural_ops["TimeBefore"],
            "after": neural_ops["TimeAfter"],
            "next": neural_ops["TimeNext"],
        }
        predefine = {
            "get_param_name_list": lambda x: x.argnames,
            "get_placeholder_list": get_placeholder_list,
            "signature": inspect.signature,
        }
        functions = dict(**neural_ops, **alias, **predefine)
        super().__init__(usersyms=dict(**variables, **functions))


class SamplingParser(BasicParser):
    def __init__(self, entity_ids: List[int], relation_ids: List[int], timestamp_ids: List[int],
                 srt2o: Dict[int, Dict[int, Dict[int, Set[int]]]],  # srt->o | entity,relation,timestamp->entity
                 sro2t: Dict[int, Dict[int, Dict[int, Set[int]]]],  # sro->t | entity,relation,entity->timestamp
                 ):
        # example
        # qe = Pe(e,r,after(Pt(e,r,e)))
        # [eid,rid,eid,rid,eid] = qe(e,r,e,r,e)
        # answers = qe(eid,rid,eid,rid,eid)
        # embedding = qe(eid,rid,eid,rid,eid)
        all_entity_ids = set(entity_ids)
        all_timestamp_ids = set(timestamp_ids)
        max_timestamp_id = max(timestamp_ids)
        # print(srt2o)
        # print(sro2t)

        variables = {
            "e": Placeholder("e"),
            "r": Placeholder("r"),
            "t": Placeholder("t"),
        }
        for e_id in entity_ids:
            variables[f"e{e_id}"] = FixedQuery(answers={e_id}, is_anchor=True)
        for r_id in relation_ids:
            variables[f"r{r_id}"] = FixedQuery(answers={r_id}, is_anchor=True)
        for t_id in timestamp_ids:
            variables[f"t{t_id}"] = FixedQuery(timestamps={t_id}, is_anchor=True)

        def sampling_one_entity():
            entity = random.choice(list(srt2o.keys()))
            # print("sampling_entity", entity)
            return entity

        def sampling_one_relation_for_s(s: Set[int]):
            si = random.choice(list(s))
            relation = random.choice(list(srt2o[si].keys()))
            # print("sampling_relation_for_s", relation)
            return relation

        def sampling_one_timestamp_for_sr(s: Set[int], r: Set[int]):
            si = random.choice(list(s))
            rj = random.choice(list(r))
            timestamps = random.choice(list(srt2o[si][rj].keys()))
            # print("sampling_timestamp_for_sr", timestamps)
            return timestamps

        def sampling_one_entity_for_sr(s: Set[int], r: Set[int]):
            si = random.choice(list(s))
            rj = random.choice(list(r))
            entities = random.choice(list(sro2t[si][rj].keys()))
            # print("sampling_entity_for_sr", entities)
            return entities

        def find_entity(s: Union[FixedQuery, Placeholder], r: Union[FixedQuery, Placeholder], t: Union[FixedQuery, Placeholder]):
            if isinstance(s, Placeholder):
                s.fill(sampling_one_entity())
                s = FixedQuery(answers={s.idx}, is_anchor=True)
            if isinstance(r, Placeholder):
                r.fill(sampling_one_relation_for_s(s.answers))
                r = FixedQuery(answers={r.idx}, is_anchor=True)
            if isinstance(t, Placeholder):
                t.fill(sampling_one_timestamp_for_sr(s.answers, r.answers))
                t = FixedQuery(timestamps={t.idx}, is_anchor=True)
            answers = set()
            for si in s.answers:
                for rj in r.answers:
                    for tk in t.timestamps:
                        answers = answers | set(srt2o[si][rj][tk])
            # print("find_entity", answers)
            return answers

        def find_timestamp(s: Union[FixedQuery, Placeholder], r: Union[FixedQuery, Placeholder], o: Union[FixedQuery, Placeholder]):
            if isinstance(s, Placeholder):
                s.fill(sampling_one_entity())
                s = FixedQuery(answers={s.idx}, is_anchor=True)
            if isinstance(r, Placeholder):
                r.fill(sampling_one_relation_for_s(s.answers))
                r = FixedQuery(answers={r.idx}, is_anchor=True)
            if isinstance(o, Placeholder):
                o.fill(sampling_one_entity_for_sr(s.answers, r.answers))
                o = FixedQuery(answers={o.idx}, is_anchor=True)
            timestamps = set()
            for si in s.answers:
                for rj in r.answers:
                    for ok in o.answers:
                        timestamps = timestamps | set(sro2t[si][rj][ok])
            # print("find_timestamp", timestamps)
            return timestamps

        neural_ops = {
            "AND": lambda q1, q2: FixedQuery(answers=q1.answers & q2.answers, timestamps=q1.timestamps & q2.timestamps),
            "OR": lambda q1, q2: FixedQuery(answers=q1.answers | q2.answers, timestamps=q1.timestamps & q2.timestamps),
            "NOT": lambda x: FixedQuery(answers=all_entity_ids - x.answers, timestamps=x.timestamps),
            "EntityProjection": lambda s, r, t: FixedQuery(answers=find_entity(s, r, t)),
            "TimeProjection": lambda s, r, o: FixedQuery(timestamps=find_timestamp(s, r, o)),
            "TimeAnd": lambda q1, q2: FixedQuery(answers=q1.answers & q2.answers, timestamps=q1.timestamps & q2.timestamps),
            "TimeOr": lambda q1, q2: FixedQuery(answers=q1.answers & q2.answers, timestamps=q1.timestamps | q2.timestamps),
            "TimeNot": lambda x: FixedQuery(answers=x.answers, timestamps=all_timestamp_ids - x.timestamps if len(x.timestamps) > 0 else set()),
            "TimeBefore": lambda x: FixedQuery(answers=x.answers, timestamps=set([t for t in timestamp_ids if t < min(x.timestamps)] if len(x.timestamps) > 0 else all_timestamp_ids)),
            "TimeAfter": lambda x: FixedQuery(answers=x.answers, timestamps=set([t for t in timestamp_ids if t > max(x.timestamps)] if len(x.timestamps) > 0 else all_timestamp_ids)),
            "TimeNext": lambda x: FixedQuery(answers=x.answers, timestamps=set([min(t + 1, max_timestamp_id) for t in x.timestamps] if len(x.timestamps) > 0 else all_timestamp_ids)),
        }
        super().__init__(variables=variables, neural_ops=neural_ops)
        for _, qs in query_structures.items():
            self.eval(qs)


class NeuralParser(BasicParser):
    def __init__(self):
        # example
        # qe = Pe(e,r,after(Pt(e,r,e)))

        class QueryEmbedding:

            def __init__(self, name):
                print(name)
                self.query_embedding = None

        variables = {}
        neural_ops = {
            "AND": lambda q1, q2: QueryEmbedding(f"AND {(q1, q2)}"),
            "OR": lambda q1, q2: QueryEmbedding(f"OR {(q1, q2)}"),
            "NOT": lambda x: QueryEmbedding(f"NOT {x}"),
            "EntityProjection": lambda s, r, t: QueryEmbedding(f"EntityProjection {(s, r, t)}"),
            "TimeProjection": lambda s, r, o: QueryEmbedding(f"TimeProjection {(s, r, o)}"),
            "TimeAnd": lambda q1, q2: QueryEmbedding(f"TimeAnd {(q1, q2)}"),
            "TimeOr": lambda q1, q2: QueryEmbedding(f"TimeOr {(q1, q2)}"),
            "TimeNot": lambda x: QueryEmbedding(f"TimeNot {x}"),
            "TimeBefore": lambda x: QueryEmbedding(f"TimeBefore {x}"),
            "TimeAfter": lambda x: QueryEmbedding(f"TimeAfter {x}"),
            "TimeNext": lambda x: QueryEmbedding(f"TimeNext {x}"),
        }
        super().__init__(variables=variables, neural_ops=neural_ops)
        self.ast_cache = {}
