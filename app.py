import argparse
import numpy as np
from algorithm import Surfing


parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', help='Path to where the dataset is located')
parser.add_argument('-d', '--delimiter', default=',',
					help='Change dataset parser delimiter definition')
parser.add_argument('-s', '--skip-header', action='store_true',
					help='Determine if the dataset parser should skip a header (i.e. if the file has a header)')

if __name__ == '__main__':
	args = parser.parse_args()
	header = 1 if args.skip_header else 0
	dataset = np.genfromtxt(args.dataset_path, delimiter=args.delimiter, skip_header=header)
	unlabeled_dataset = dataset[:,:-1]

	#unlabeled_dataset = np.random.uniform(0, 1, (1000, 4))

	model = Surfing(k=3)
	model.fit(unlabeled_dataset)
	