

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

