#!/usr/bin/env python

import numpy as np
from scipy.special import jn
import matplotlib.pyplot as plt


def main():
    array_width = 20
    l0 = 1

    npts1d = 10
    npts = npts1d**2
    x = np.linspace(-array_width/2, array_width/2, npts1d)
    y = np.linspace(-array_width/2, array_width/2, npts1d)
    dx = x[1] - x[0]
    xgrid, ygrid = np.meshgrid(x, y, indexing='ij')
    xs = xgrid.flatten()
    ys = ygrid.flatten()

    distances = np.sqrt((xs[None, :] - xs[:, None])**2 +
                        (ys[:, None] - ys[None, :])**2)
    # covariance = jn(0, 2. * np.pi * distances)
    covariance = np.exp(-distances**2 / 5.)

    covariancex = np.exp(-(x[None, :] - x[:, None])**2/5.)
    covariancey = np.exp(-(y[None, :] - y[:, None])**2/5.)
    covariance_combined = np.einsum('ij, kl -> ikjl', covariancex, covariancey)
    covariance_combined = covariance_combined.reshape(npts, npts)

    fig, (col1, col2, col3) = plt.subplots(1, 3)
    col1.imshow(covariancex)
    col2.imshow(covariancey)
    col3.imshow(covariance_combined, cmap='viridis')

    evals, evecs = np.linalg.eigh(covariance)

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
    plt.show()

    # Computed from Hankel (Fourier-Bessel) transform
    rho = np.sqrt(xgrid**2 + ygrid**2)
    kx = np.linspace(-np.pi/dx, np.pi/dx)
    kxgrid, kygrid = np.meshgrid(kx, kx, indexing='ij')
    kabs = np.sqrt(kxgrid**2 + kygrid**2)

    # n = 9
    # alpha_0 = besselzero(n, nx*nx, 1)
    #
    # 
    #c = zeros(1, nx*nx) ;
    #
    # 
    #drho = diff(rho) ;
    #rho = rho[1:-1] ;
    #for ii = 1:nx*nx
    #    cdrho = k0^2 * (besselj(0, rho*k0)) .* besselj(n, alpha_0(ii)*k0*rho/L) .* rho .* drho ;
    #    c(ii) = sum(cdrho) ;
    #    c(ii) = 2*c(ii)/(L^2*besselj(n+1, alpha_0(ii))^2);
    #end
    #phi = c ;
    #
    #figure(2)
    #clf
    #plot(lambdas, '.-')
    #plot(phi, '.-')
    #box on


if __name__ == "__main__":
    main()
