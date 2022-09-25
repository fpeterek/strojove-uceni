import click

from apriori import gen_combinations as gen_combinations_int
from apriori import find_patterns as find_patterns_int


@click.command()
@click.option('--begin', help='Start of interval')
@click.option('--end', help='End of interval')
@click.option('--elements', help='Minimum confidence')
def gen_combinations(begin: int, end: int, elements: int) -> None:
    combinations = gen_combinations_int(int(begin), int(end), int(elements))
    for c in combinations:
        print(c)


@click.command()
@click.option('--file', help='Path to dataset')
@click.option('--min-sup', default=0.25, help='Minimum support')
@click.option('--min-conf', default=0.5, help='Minimum confidence')
def find_patterns(file: str, min_sup: float, min_conf: float) -> None:
    find_patterns_int(file, float(min_sup), float(min_conf))


@click.group('Apriori')
def main() -> None:
    pass


main.add_command(find_patterns)
main.add_command(gen_combinations)


if __name__ == '__main__':
    main()

