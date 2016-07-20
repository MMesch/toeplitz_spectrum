#!/usr/bin/env python

import numpy as np
from scipy.special import jn
import matplotlib.pyplot as plt


def main():
    # get covariance matrix
    # covariance = get_2dcovariance(nx=20, ny=20)
    covariance = get_1dcovariance(nx=50)

    npts, _ = covariance.shape
    # directly compute eigenvectors/values
    evals, evecs = np.linalg.eigh(covariance)

    plot_spectrum(covariance, evals, evecs)

    analytic_toeplitz_spectrum(covariance)

    plt.show()


def analytic_toeplitz_spectrum(tmatrix):
    npts, _ = tmatrix.shape

    # check if toeplitz:
    # indices = np.arange(npts)
    # for ipts in range(npts):
    #    idx1 = indices[:-ipts]
    #    idx2 = (indices + ipts)[:-ipts]

    


def plot_spectrum(covariance, evals, evecs):
    npts, _ = covariance.shape
    # plot direct eigenvector/value computation
    fig, (col1, col2) = plt.subplots(1, 2)
    col1.imshow(covariance, aspect='auto', origin='upper', cmap='viridis',
                interpolation='nearest')
    col1.set(xlabel='r1', ylabel='r2', title='covariance matrix')
    delta = 1. / (2. * npts)
    extent = (0. - delta, 1. + delta, 0., 1.)
    col2.imshow(evecs, origin='lower', aspect='auto', cmap='RdBu',
                extent=extent)
    col2.plot(np.linspace(0., 1., npts), evals/evals.max())
    col2.set(xlabel='eigenvector', ylabel='eigenvalue or spatial coordinate',
             xlim=(0, 1), ylim=(0, 1), title='eigenvectors and values')


def get_1dcovariance(nx=20):
    array_width = 20
    x = np.linspace(-array_width/2, array_width/2, nx)

    # compute covariance matrix
    distances = x[:, None] - x[None, :]
    covariance = jn(0, distances)
    return covariance


def get_2dcovariance(nx=20, ny=20):
    array_width = 20
    x = np.linspace(-array_width/2, array_width/2, nx)
    y = np.linspace(-array_width/2, array_width/2, ny)
    xgrid, ygrid = np.meshgrid(x, y, indexing='ij')
    xs = xgrid.flatten()
    ys = ygrid.flatten()

    # compute covariance matrix
    distances = np.sqrt((xs[None, :] - xs[:, None])**2 +
                        (ys[:, None] - ys[None, :])**2)
    covariance = jn(0, distances)
    return covariance


if __name__ == "__main__":
    main()
