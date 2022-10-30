import random

import click


@click.command()
@click.option('--dataset', type=str, help='Dataset to sample')
@click.option('--output', type=str, help='Output file')
@click.option('--rate', type=float, default=0.05, help='Sampling rate')
def sample(dataset: str, output: str, rate: float) -> None:
    with open(dataset) as file:
        with open(output, 'w') as out:
            header = file.readline()
            out.write(header)
            for line in file:
                if random.random() <= rate:
                    out.write(line)


@click.group('Dataset sampling')
def main() -> None:
    pass


main.add_command(sample)


if __name__ == '__main__':
    main()
