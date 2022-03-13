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
    # 1. 1-hop Pe and Pt, manually
    # 2. entity multi-hop
    "Pe2": "def Pe2(e1, r1, t1, r2, t2): return Pe(Pe(e1, r1, t1), r2, t2)",  # 2p
    "Pe3": "def Pe3(e1, r1, t1, r2, t2, r3, t3): return Pe(Pe(Pe(e1, r1, t1), r2, t2), r3, t3)",  # 3p
    # 3. time multi-hop
    "Pt_lPe": "def Pt_lPe(e1, r1, t1, r2, e2): return Pt(Pe(e1, r1, t1), r2, e2)",  # l for left (as head entity)
    "Pt_rPe": "def Pt_rPe(e1, r1, e2, r2, t1): return Pt(e1, r1, Pe(e2, r2, t1))",  # r for right (as tail entity)
    "Pe_Pt": "def Pe_Pt(e1, r1, e2, r2, e3): return Pe(e1, r1, Pt(e2, r2, e3))",  # at
    "Pe_aPt": "def Pe_aPt(e1, r1, e2, r2, e3): return Pe(e1, r1, after(Pt(e2, r2, e3)))",  # a for after
    "Pe_bPt": "def Pe_bPt(e1, r1, e2, r2, e3): return Pe(e1, r1, before(Pt(e2, r2, e3)))",  # b for before
    "Pe_nPt": "def Pe_nPt(e1, r1, e2, r2, e3): return Pe(e1, r1, next(Pt(e2, r2, e3)))",  # n for next
    # 4. entity and & time and
    "e2i": "def e2i(e1, r1, t1, e2, r2, t2): return And(Pe(e1, r1, t1), Pe(e2, r2, t2))",  # 2i
    "e3i": "def e3i(e1, r1, t1, e2, r2, t2, e3, r3, t3): return And3(Pe(e1, r1, t1), Pe(e2, r2, t2), Pe(e3, r3, t3))",  # 3i
    "t2i": "def t2i(e1, r1, e2, e3, r2, e4): return TimeAnd(Pt(e1, r1, e2), Pt(e3, r2, e4))",  # t-2i
    "t3i": "def t3i(e1, r1, e2, e3, r2, e4, e5, r3, e6): return TimeAnd3(Pt(e1, r1, e2), Pt(e3, r2, e4), Pt(e5, r3, e6))",  # t-3i
    # 5. complex time and
    "e2i_Pe": "def e2i_Pe(e1, r1, t1, r2, t2, e2, r3, t3): return And(Pe(Pe(e1, r1, t1), r2, t2), Pe(e2, r3, t3))", # pi
    "Pe_e2i": "def Pe_e2i(e1, r1, t1, e2, r2, t2): return Pe(And(Pe(e1, r1, t1), Pe(e2, r2, t2)), r3, t3)", # ip
    "Pt_le2i": "def Pt_le2i(e1, r1, t1, e2, r2, t2, r3, e3): return Pt(e2i(e1, r1, t1, e2, r2, t2), r3, e3)",  # mix ip
    "Pt_re2i": "def Pt_re2i(e1, r1, e2, r2, t1, e3, r3, t2): return Pt(e1, r1, e2i(e2, r2, t1, e3, r3, t2))",  # mix ip
    "t2i_Pe": "def t2i_Pe(e1, r1, t1, r2, t2, e2, r3, t3): return TimeAnd(Pt(Pe(e1, r1, t1), r2, e2), Pt(e3, r3, e4))", # t-pi
    "Pe_t2i": "def Pe_t2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, t2i(e2, r2, e3, e4, r3, e5))",  # t-ip
    "Pe_at2i": "def Pe_at2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, after(t2i(e2, r2, e3, e4, r3, e5)))",
    "Pe_bt2i": "def Pe_bt2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, before(t2i(e2, r2, e3, e4, r3, e5)))",
    "Pe_nt2i": "def Pe_nt2i(e1, r1, e2, r2, e3, e4, r3, e5): return Pe(e1, r1, next(t2i(e2, r2, e3, e4, r3, e5)))",
    "between": "def between(e1, r1, e2, e3, r2, e4): return TimeAnd(after(Pt(e1, r1, e2)), before(Pt(e3, r2, e4))))",  # between t1, t2 == after t1 and before t2
    # 5. entity not
    "e2i_NPe": "def e2i_NPe(e1, r1, t1, r2, t2, e2, r3, t3): return And(Not(Pe(Pe(e1, r1, t1), r2, t2)), Pe(e2, r3, t3))",  # pni = e2i_N(Pe(e1, r1, t1), r2, t2, e2, r3, t3)
    "e2i_PeN": "def e2i_PeN(e1, r1, t1, r2, t2, e2, r3, t3): return And(Pe(Pe(e1, r1, t1), r2, t2), Not(Pe(e2, r3, t3)))",  # pin
    "Pe_e2i_Pe_NPe": "def Pe_e2i_Pe_NPe(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(And(Pe(e1, r1, t1), Not(Pe(e2, r2, t2))), r3, t3)",  # inp
    "e2i_N": "def e2i_N(e1, r1, t1, e2, r2, t2): return And(Pe(e1, r1, t1), Not(Pe(e2, r2, t2)))",  # 2in
    "e3i_N": "def e3i_N(e1, r1, t1, e2, r2, t2, e3, r3, t3): return And3(Pe(e1, r1, t1), Pe(e2, r2, t2), Not(Pe(e3, r3, t3)))",  # 3in
    # 6. time not
    "t2i_NPt": "def t2i_NPt(e1, r1, t1, r2, t2, e2, r3, t3): return TimeAnd(TimeNot(Pt(Pe(e1, r1, t1), r2, e2)), Pt(e3, r3, e4)))",  # t-pni
    "t2i_PtN": "def t2i_PtN(e1, r1, t1, r2, e2, e3, r3, e4): return TimeAnd(Pt(Pe(e1, r1, t1), r2, e2), TimeNot(Pt(e3, r3, e4)))",  # t-pin
    "Pe_t2i_PtPe_NPt": "def Pe_t2i_PtPe_NPt(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(e1, r1, TimeAnd(Pt(Pe(e1, r1, t1), r1, e2), TimeNot(Pt(e3, r2, e4))))",  # t-inp
    "t2i_N": "def t2i_N(e1, r1, e2, e3, r2, e4): return TimeAnd(Pt(e1, r1, e2), TimeNot(Pt(e3, r2, e4)))",  # t-2in
    "t3i_N": "def t3i_N(e1, r1, e2, e3, r2, e4, e5, r3, e6): return TimeAnd3(Pt(e1, r1, e2), Pt(e3, r2, e4), TimeNot(Pt(e5, r3, e6)))",  # t-3in
    # 7. entity union & time union
    "e2u": "def e2u(e1, r1, t1, e2, r2, t2): return Or(Pe(e1, r1, t1), Pe(e2, r2, t2))",  # 2u
    "Pe_e2u": "def Pe_e2u(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(Or(Pe(e1, r1, t1), Pe(e2, r2, t2)), r3, t3)",  # up
    "t2u": "def t2u(e1, r1, e2, e3, r2, e4): return TimeOr(Pt(e1, r1, e2), Pt(e3, r2, e4))",  # t-2u
    "Pe_t2u": "def Pe_t2u(e1, r1, e2, e3, r2, e4, e5, r3, e6): return Pe(e1, r1, TimeOr(Pt(e2, r2, e3), Pt(e4, r3, e5))",  # t-up
    # 8. union-DM
    "e2u_DM": "def e2u_DM(e1, r1, t1, e2, r2, t2): return And(Not(Pe(e1, r1, t1)), Not(Pe(e2, r2, t2)))",  # 2u-DM
    "Pe_e2u_DM": "def Pe_e2u_DM(e1, r1, t1, e2, r2, t2, r3, t3): return Pe(And(Not(Pe(e1, r1, t1)), Not(Pe(e2, r2, t2))), r3, t3)",  # up-DM
    "t2u_DM": "def t2u_DM(e1, r1, e2, e3, r2, e4): return TimeAnd(TimeNot(Pt(e1, r1, e2)), TimeNot(Pt(e3, r2, e4)))",  # t-2u-DM
    "Pe_t2u_DM": "def Pe_t2u_DM(e1, r1, e2, e3, r2, e4, e5, r3, e6): return Pe(e1, r1, TimeAnd(TimeNot(Pt(e2, r2, e3)), TimeNot(Pt(e4, r3, e5)))",  # t-up-DM
}
train_query_structures = [
    # entity
    "Pe", "Pe2", "Pe3", "e2i", "e3i",  # 1p, 2p, 3p, 2i, 3i
    "e2i_NPe", "e2i_PeN", "Pe_e2i_Pe_NPe", "e2i_N", "e3i_N",  # npi, pni, inp, 2in, 3in
    # time
    "Pt", "Pt_lPe", "Pt_rPe", "Pe_Pt", "Pe_aPt", "Pe_bPt", "Pe_nPt",  # t-1p, t-2p
    "t2i", "t3i", "Pt_le2i", "Pt_re2i", "Pe_t2i", "Pe_at2i", "Pe_bt2i", "Pe_nt2i", "between",  # t-2i, t-3i
    "t2i_NPt", "t2i_PtN", "Pe_t2i_PtPe_NPt", "t2i_N", "t3i_N",  # t-npi, t-pni, t-inp, t-2in, t-3in
]
test_query_structures = train_query_structures + [
    # entity
    "e2i_Pe", "Pe_e2i", # pi, ip
    "e2u", "Pe_e2u",  # 2u, up
    # time
    "t2i_Pe", "Pe_t2i", # t-pi, t-ip
    "t2u", "Pe_t2u",  # t-2u, t-up
]


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
            "And": lambda q1, q2: FixedQuery(answers=q1.answers & q2.answers, timestamps=q1.timestamps & q2.timestamps),
            "And3": lambda q1, q2, q3: FixedQuery(answers=q1.answers & q2.answers & q3.answers, timestamps=q1.timestamps & q2.timestamps & q3.timestamps),
            "Or": lambda q1, q2: FixedQuery(answers=q1.answers | q2.answers, timestamps=q1.timestamps & q2.timestamps),
            "Not": lambda x: FixedQuery(answers=all_entity_ids - x.answers, timestamps=x.timestamps),
            "EntityProjection": lambda s, r, t: FixedQuery(answers=find_entity(s, r, t)),
            "TimeProjection": lambda s, r, o: FixedQuery(timestamps=find_timestamp(s, r, o)),
            "TimeAnd": lambda q1, q2: FixedQuery(answers=q1.answers & q2.answers, timestamps=q1.timestamps & q2.timestamps),
            "TimeAnd3": lambda q1, q2, q3: FixedQuery(answers=q1.answers & q2.answers & q3.answers, timestamps=q1.timestamps & q2.timestamps & q3.timestamps),
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
