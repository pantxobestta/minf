"""Core utilities for the minf package."""

import numpy as np


def mutual_information(a, b, bins=256):
    """
    Compute the mutual information between two equally-sized 1-D signals or
    2-D images using a fast joint-histogram approach.

    Parameters
    ----------
    a : array_like
        First signal or image. Can be any shape; it will be flattened.
    b : array_like
        Second signal or image. Must have the same number of elements as ``a``.
    bins : int, optional
        Number of bins to use for the joint histogram. Default is 256.

    Returns
    -------
    float
        The estimated mutual information in nats.

    Notes
    -----
    The implementation follows the algorithm described in the MATLAB
    FileExchange submission (Fast mutual information of two images or signals).

    * The joint histogram is computed with ``np.histogram2d``.
    * Mutual information is estimated as
      ``MI = Σ p_{ij} * log(p_{ij} / (p_i * p_j))``,
      where ``p_{ij}`` is the joint probability and ``p_i``, ``p_j`` are the
      marginal probabilities.
    * Zero probabilities are ignored (``0 * log(0)`` is treated as ``0``).

    Examples
    --------
    >>> import numpy as np
    >>> # Two identical constant signals
    >>> x = np.array([1, 1, 1, 1])
    >>> mutual_information(x, x)
    0.0
    """
    # Ensure inputs are numpy arrays and have the same size
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size:
        raise ValueError("Inputs a and b must contain the same number of elements.")

    # Compute joint histogram
    hist, _, _ = np.histogram2d(a, b, bins=bins)

    # Convert counts to probabilities
    joint_prob = hist / hist.size  # flatten and divide by total count
    # Marginal probabilities
    p_a = joint_prob.sum(axis=1)  # sum over columns -> shape (bins,)
    p_b = joint_prob.sum(axis=0)  # sum over rows    -> shape (bins,)

    # Compute mutual information
    # Avoid log(0) by masking out zero entries
    mask = joint_prob > 0
    mi = np.sum(
        joint_prob[mask] * np.log(
            joint_prob[mask] / (p_a[None, :][mask] * p_b[:, None][mask])
        )
    )
    return mi