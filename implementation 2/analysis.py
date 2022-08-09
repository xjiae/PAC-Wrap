import numpy as np

import scipy.stats
import math
import random
import argparse

def clopper_pearson(x, n, alpha=0.05):
    """Estimate the confidence interval for a sampled Bernoulli random
    variable.
    `x` is the number of successes and `n` is the number trials (x <=
    n). `alpha` is the confidence level (i.e., the true probability is
    inside the confidence interval with probability 1-alpha). The
    function returns a `(low, high)` pair of numbers indicating the
    interval on the probability.
    """
    b = scipy.stats.beta.ppf
    lo = b(alpha / 2.0, x, n - x + 1)
    hi = b(1 - alpha / 2.0, x + 1, n - x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", action="store_true", help="run synthetic dataset or benchmark dataset")
    # parser.add_argument("--detector", type=str, default="IF", help="benchmark dataset path")
    args = parser.parse_args()

    # detector = args.detector
    synth = args.synth
    detectors = ['LOF']
    for detector in detectors:
        print('detector', detector)
        synth = args.synth
        results = np.load(f'{detector}_results_{synth}.npy', allow_pickle=True)
        trials = 4000
        data = results.reshape([-1, 6])
        batch_size = 25
        epsilon = 0.05
        iters = int(data.shape[0] / batch_size)

        total = results.shape[0]
        successes = results > epsilon
        rates = ['0.5 fpr', '0.75 fpr', '1 fpr', '0.5 fnr', '0.75 fnr', '1 fnr']
        print('Anomaly detector', detector)
        print('Is synthetic dataset', synth)
        for i in range(6):
            lo, hi = clopper_pearson(np.sum(successes[:, i]), total)
            print('{:s} 95% confidence interval: {:.3f}-{:.3f}'.format(rates[i], lo, hi))
