# crosci: biomarkers of Critical Oscillations

Collection of biomarkers for estimating criticality of neuronal oscillations:

- detrended fluctuation analysis (DFA), as implemented in [1]
- fE/I, a non-invasive estimate of E/I ratio [2]

The spectral ranges for computation of DFA and fE/I have been optimized in [3]

To reference, please cite the following papers:

1. Hardstone, Richard, et al. "Detrended fluctuation analysis: a scale-free view on neuronal oscillations." Frontiers in physiology 3 (2012): 450.
2. Bruining, Hilgo, et al. "Measurement of excitation-inhibition ratio in autism spectrum disorder using critical brain dynamics." Scientific reports 10.1 (2020): 9195.
3. Diachenko, Marina, et al. "Functional excitation-inhibition ratio indicates near-critical oscillations across frequencies." Imaging Neuroscience 2 (2024): 1-17.

## Running

The demo script (demo.py) exemplifies how DFA and fEI are computed for randomly generated data. You can use it as a starting point for your own data.

## Building

We use uv and setuptools to build this package. As we also use Cyhton to improve performance you will need to have clang installed on your system

To build go into the source directory and do

```bash
uv build
```

This will both build a SDist and a wheel of the package for your machine.

### Note

Crosci is also dependent on openmp, with is missing on modern MacOS clang version Apple ships. For the build to be succesfull you will need to install libomp as shown below:

```zsh
brew install libomp
```
