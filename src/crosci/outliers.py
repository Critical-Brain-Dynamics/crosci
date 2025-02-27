# code adapted from PyAstronomy.pyasl.asl.outlier.generalizedESD, licensed under the MIT license https://opensource.org/license/mit
from scipy.stats import t
import numpy as np
import numpy.ma as ma


def generalizedESD(x, maxOLs, alpha=0.05, fullOutput=False, ubvar=False):
    """
    Carry out a Generalized ESD Test for Outliers.

    The Generalized Extreme Studentized Deviate (ESD) test for
    outliers can be used to search for outliers in a univariate
    data set, which approximately follows a normal distribution.
    A description of the algorithm is, e.g., given at
    `Nist <http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm>`_
    or [Rosner1983]_.

    Parameters
    ----------
    maxOLs : int
        Maximum number of outliers in the data set.
    alpha : float, optional
        Significance (default is 0.05).
    fullOutput : boolean, optional
        Determines whether additional return values
        are provided. Default is False.
    ubvar : boolean, optional
        If True, an unbiased estimate for the variance will be used in the
        calculation; this provides compatibility with the R implementation of
        NIST. If False (default), the maximum-likelihood variance estimate
        will be used.

    Returns
    -------
    Number of outliers : int
        The number of data points characterized as
        outliers by the test.
    Indices : list of ints
        The indices of the data points found to
        be outliers.
    R : list of floats, optional
        The values of the "R statistics". Only provided
        if `fullOutput` is set to True.
    L : list of floats, optional
        The lambda values needed to test whether a point
        should be regarded an outlier. Only provided
        if `fullOutput` is set to True.

    """
    if maxOLs < 1:
        raise ValueError(
            "Maximum number of outliers, `maxOLs`, must be > 1. "
            "Specify, e.g., maxOLs = 2."
        )
    xm = ma.array(x)
    n = len(xm)
    # Compute R-values
    R = []
    L = []
    minds = []
    for i in range(maxOLs + 1):
        # Compute mean and std of x
        xmean = xm.mean()
        xstd = xm.std(ddof=int(ubvar))
        # Find maximum deviation
        rr = np.abs((xm - xmean) / xstd)
        minds.append(np.argmax(rr))
        R.append(rr[minds[-1]])
        if i >= 1:
            p = 1.0 - alpha / (2.0 * (n - i + 1))
            perPoint = t.ppf(p, n - i - 1)
            L.append(
                (n - i) * perPoint / np.sqrt((n - i - 1 + perPoint**2) * (n - i + 1))
            )
        # Mask that value and proceed
        xm[minds[-1]] = ma.masked
    # Remove the first entry from R, which is of
    # no meaning for the test
    R.pop(-1)
    # Find the number of outliers
    ofound = False
    for i in range(maxOLs - 1, -1, -1):
        if R[i] > L[i]:
            ofound = True
            break
    # Prepare return value
    if ofound:
        if not fullOutput:
            # There are outliers
            return i + 1, minds[0 : i + 1]
        else:
            return i + 1, minds[0 : i + 1], R, L, minds
    else:
        # No outliers could be detected
        if not fullOutput:
            # There are outliers
            return 0, []
        else:
            return 0, [], R, L, minds
