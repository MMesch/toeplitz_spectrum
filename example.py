#!/usr/bin/env python

import numpy as np
from scipy.special import jn
import matplotlib.pyplot as plt


def main():
    array_width = 20
    npts1d = 20
    npts = npts1d**2
    x = np.linspace(-array_width/2, array_width/2, npts1d)
    y = np.linspace(-array_width/2, array_width/2, npts1d)
    xgrid, ygrid = np.meshgrid(x, y, indexing='ij')
    xs = xgrid.flatten()
    ys = ygrid.flatten()

    # compute covariance matrix
    distances = np.sqrt((xs[None, :] - xs[:, None])**2 +
                        (ys[:, None] - ys[None, :])**2)
    covariance = jn(0, distances)

    # directly compute eigenvectors/values
    evals, evecs = np.linalg.eigh(covariance)

    # plot direct eigenvector/value computation
    fig, (col1, col2) = plt.subplots(1, 2)
    col1.imshow(covariance, aspect='auto', origin='upper', cmap='viridis')
    col1.set(xlabel='r1', ylabel='r2', title='covariance matrix')
    delta = 1./(2.*npts)
    extent = (0. - delta, 1. + delta, 0., 1.)
    col2.imshow(evecs, origin='lower', aspect='auto', cmap='RdBu',
                extent=extent)
    col2.plot(np.linspace(0., 1., npts), evals/evals.max())
    col2.set(xlabel='eigenvector', ylabel='eigenvalue or spatial coordinate',
             xlim=(0, 1), ylim=(0, 1), title='eigenvectors and values')

    # rearange toeplitz matrix into associated form
    covariance_new = covariance.reshape(npts1d, npts1d, npts1d, npts1d)
    covariance_new = np.transpose(covariance_new, axes=(2, 3, 0, 1))
    covariance_new = covariance_new.reshape(npts, npts)
    fig, (col1, col2) = plt.subplots(1, 2)
    col1.imshow(covariance_new, aspect='auto', origin='upper', cmap='viridis')
    col1.set(xlabel='r1', ylabel='r2', title='covariance matrix')


    plt.show()


if __name__ == "__main__":
    main()
