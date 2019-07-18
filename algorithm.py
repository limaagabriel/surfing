import numpy as np


class Surfing(object):
	def __init__(self, k=3):
		self.__k = k

	def fit(self, x):
		tau, subspaces = self.__fit_one_dimensional_spaces(x)

	def __fit_one_dimensional_spaces(self, x):
		subspaces = x.transpose()
		qualities = self.__quality_measure(subspaces)
		sl, sh = qualities.min(), qualities.max()

		tau = sh / 2.0
		if sl <= (2.0 * sh) / 3.0:
			tau = sl
			subspaces = np.delete(subspaces, qualities.argmin())

		return tau, subspaces

	def __quality_measure(self, subspaces):
		def dist_fn(subspace):
			output = np.zeros_like(subspace)
			for idx, p in enumerate(subspace):
				subspace_without_p = np.delete(subspace, idx)
				neighborhood = self.__knn(p, subspace_without_p)
				output[idx] = neighborhood.max()
				
			return (output - output.min()) / (output.max() - output.min())

		distances = np.apply_along_axis(dist_fn, 1, subspaces)
		mean_s = distances.mean(axis=1).reshape(-1, 1)
		diff_s = 0.5 * np.sum(np.abs(mean_s - distances), axis=1).reshape(-1, 1)
		qualities = np.zeros_like(diff_s)

		for idx, _ in enumerate(subspaces):
			n_below_s = np.sum(distances[idx, :] < mean_s[idx])
			if n_below_s == 0:
				qualities[idx] = 0
				continue

			qualities[idx] = diff_s[idx] / (n_below_s * mean_s[idx])
		return qualities


	def __knn(self, p, x):
		x = x.reshape(x.shape[0], -1)
		distances = np.sqrt(((x - p) ** 2).sum(axis=1))
		idx = np.argsort(distances)[:self.__k]
		return distances[idx]