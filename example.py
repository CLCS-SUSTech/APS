import os
from nll import NLLCal
from autoperiod import AutoPeriod
import numpy as np


def text_to_nll(text_path, model_path):
    with open(text_path, 'r') as f:
        text = f.read().strip()
    texts = [text]

    nll_cal = NLLCal(model_path=model_path)
    nll = nll_cal.texts_to_nll(texts)[0].numpy()
    return nll


def main():
    # nll = text_to_nll('wsj_2380.txt', '/path/to/your/model')
    # # save nll to file
    # np.save('wsj_2380_nll.npy', nll)

    nll = np.load('wsj_2380_nll.npy')

    times = np.arange(len(nll))
    # run autoperoid
    results = AutoPeriod(
        times,
        nll,
        mc_iterations=1000,
        confidence_level=.99,
    ).run()
    print(results)


if __name__ == "__main__":
    main()