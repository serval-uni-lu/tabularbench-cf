# TabularBench

TabularBench: Adversarial Robustness Benchmark for Tabular Data

This is a fork from the [original benchmark repository](https://github.com/serval-uni-lu/tabularbench).
This version has been updated to work with `python 3.12` and the latest versions of the dependencies.
It has also been refactored to be more lightweight, and thus the original leaderbord and documentation have been removed.
To access the original leaderboard and documentation, please refer to the original repository.

**Research papers**:

- Benchmark: [TabularBench: Benchmarking Adversarial Robustness for Tabular Deep Learning in Real-world Use-cases](https://arxiv.org/abs/2408.07579)
- CAA attack: [Constrained Adaptive Attack: Effective Adversarial Attack Against Deep Neural Networks for Tabular Data](https://arxiv.org/abs/2406.00775)
- CAPGD attack: [Towards Adaptive Attacks on Constrained Tabular Machine Learning](https://openreview.net/forum?id=DnvYdmR9OB)

**How to cite**:

To reference the *CAA attack*, consider citing the following paper:

```bibtex
@misc{simonetto2024caa,
    title={Constrained Adaptive Attack: Effective Adversarial Attack Against Deep Neural Networks for Tabular Data},
    author={Thibault Simonetto and Salah Ghamizi and Maxime Cordy},
    booktitle={To appear in Advances in Neural Information Processing Systems},
    year={2024},
    url={https://arxiv.org/abs/2406.00775},
}
```

To reference the benchmark, consider citing the following paper:

```bibtex
@misc{simonetto2024tabularbench,
    title={TabularBench: Benchmarking Adversarial Robustness for Tabular Deep Learning in Real-world Use-cases},
    author={Thibault Simonetto and Salah Ghamizi and Maxime Cordy},
    booktitle={To appear in Advances in Neural Information Processing Systems},
    year={2024},
    url={https://arxiv.org/abs/2408.07579},
}
```


## Installation

### UV

1. Clone the repository

``bash
git clone https://github.com/serval-uni-lu/tabularbench-cf.git
``

2. Navigate to the project directory

```bash
cd tabularbench-cf
```

3. Install the dependencies using [UV](https://uv.io/).

```bash
uv sync
```