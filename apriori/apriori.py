from typing import Any


def gen_combinations(begin: int, end: int, elems: int) -> list[list[int]]:

    all_combinations = []

    if elems <= 0:
        return all_combinations

    # Python ranges are exclusive, end param of this function is inclusive, thus we add +1
    # Combinations of one element (with begin == end) need to actually contain that one element,
    # begin must not equal end, hence we add another +1
    for i in range(begin, end+2-elems):
        start = [i]
        combinations = gen_combinations(i+1, end, elems-1)
        all_combinations += [start + c for c in combinations] if combinations else [start]

    return all_combinations


def get_combinations(lst: list[Any], elems: int, offset: int = 0) -> list[list[Any]]:

    all_combinations = []

    if elems <= 0:
        return all_combinations

    for i in range(offset, len(lst)+1-elems):
        start = [lst[i]]
        combinations = get_combinations(lst, elems-1, i+1)
        all_combinations += [start + c for c in combinations] if combinations else [start]

    return all_combinations


def load_ds(file: str) -> list[set[int]]:
    with open(file) as f:
        lines = f.readlines()
    return [{int(record) for record in line.split(' ')} for line in lines]


def create_support_table(ds: list[set[int]], min_supp: float) -> dict[int, int]:
    supports = dict()

    for record in ds:
        for value in record:
            supports[value] = supports.get(value, 0) + 1

    return {value: count/len(ds) for value, count in supports.items() if count/len(ds) >= min_supp}


def get_init_fis(ds: list[set[int]], min_supp: float) -> dict[int, int]:
    supp_table = create_support_table(ds, min_supp)
    return [{val} for val in supp_table]


def get_candidates(ds: list[set[int]]) -> list[set[int]]:
    sets = []
    for i in range(len(ds) - 1):
        for j in range(i+1, len(ds)):
            fst, snd = ds[i], ds[j]
            diff = fst.symmetric_difference(snd)
            if len(diff) == 2:
                sets.append(fst.union(diff))
    return sets


def filter_by_support(candidates: list[set[int]], ds: list[set[int]], supp: float) -> list[set[int]]:

    supp_table = dict()
    # Unique candidates
    candidates = [set(c) for c in {tuple(c) for c in candidates}]

    for c in candidates:
        for rec in ds:
            if rec.issuperset(c):
                as_tuple = tuple(c)
                supp_table[as_tuple] = supp_table.get(as_tuple, 0) + 1

    st = {k: c/len(ds) for k, c in supp_table.items()}

    return [set(s) for s, count in supp_table.items() if count/len(ds) > supp]


def find_fis(ds: list[set[int]], prev_fis: list[set[int]], supp: float) -> list[set[int]]:
    if not prev_fis:
        return []

    candidates = filter_by_support(get_candidates(prev_fis), ds, supp)

    rec_candidates = find_fis(ds, candidates, supp)

    return candidates + rec_candidates


def find_all_fis(ds: list[set[int]], supp: float) -> list[set[int]]:
    init_fis = get_init_fis(ds, supp)
    rest = find_fis(ds, init_fis, supp)
    return init_fis + rest


def get_rule_confidence(l: set[int], r: set[int], ds: list[set[int]]) -> float:
    l_conf = 0
    union_conf = 0
    union = l | r

    for rule in ds:
        l_conf += rule.issuperset(l)
        union_conf += rule.issuperset(union)

    l_conf /= len(ds)
    union_conf /= len(ds)

    return union_conf / l_conf


def gen_rules_for_set(fis: set[int], ds: list[set[int]], conf: float) -> set[tuple, tuple, float]:
    as_list = list(fis)
    rules = set()

    for elems in range(1, len(as_list)):
        subsets = get_combinations(as_list, elems) 
        print(f'{subsets=}')
        for subset in subsets:
            as_set = set(subset)
            l, r = as_set, fis - as_set
            rule_conf = get_rule_confidence(l, r, ds)
            if rule_conf >= conf:
                rules.add((tuple(sorted(l)), tuple(sorted(r)), rule_conf))

    return rules


def gen_rules(fis: list[set[int]], ds: list[set[int]], conf: float) -> set[tuple, tuple, float]:
    fis = [s for s in fis if len(s) > 1]
    rules = set()

    for itemset in fis:
        rules |= gen_rules_for_set(itemset, ds, conf)

    return rules


def find_patterns(file: str, supp: float, conf: float):
    ds = load_ds(file)
    fis = find_all_fis(ds, supp)
    for f in fis:
        print(f'fis: {f}')
    rules = gen_rules(fis, ds, conf)
    for rule in rules:
        l, r, conf = rule
        print(f'{l} => {r} ({conf})')

