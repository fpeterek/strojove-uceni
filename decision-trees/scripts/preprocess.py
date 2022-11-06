import sys


def main(infile, outfile):
    with open(infile) as f:
        with open(outfile, 'w') as out:
            for line in f:
                split = line.strip().split(',')
                survived = split.pop(1)
                split.append(survived)
                recon = ','.join(split)
                out.write(recon)
                out.write('\n')


if __name__ == '__main__':
    infile, outfile = sys.argv[1], sys.argv[2]
    main(infile, outfile)
