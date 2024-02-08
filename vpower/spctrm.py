#----------------------------------------------------------------------
#       File:			spctrm.py
#		Programmer:		Yujie He
#		Last modified:	22/09/23                # dd/mm/yy
#		Description:    Functions for power spectrum analysis.
#----------------------------------------------------------------------
#
#       This file contains functions related to power spectrum 
#       calculation.
#
#       Functions and classes starting with '_' are for intrinsic use 
#       only. 
#
#----------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import pyfftw


def _vector_power(fx, fy, fz, Lbox, Nsize):
    """
  Calculate FFT and power grid before sampling. This function does the main 
  math and physics in power spectrum computation.

  Default normalization is such that
  `np.sum(Pk*(2*np.pi/Lbox)**3)` and `0.5*np.mean(vx**2+vy**2+vz**2)` are equal
  """
    # Fourier transform
    a = (Lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3
    fkx = pyfftw.interfaces.numpy_fft.fftn(fx) * a
    fky = pyfftw.interfaces.numpy_fft.fftn(fy) * a
    fkz = pyfftw.interfaces.numpy_fft.fftn(fz) * a
    # Definition of velocity power spectrum
    Pk = 0.5 * (np.abs(fkx) ** 2 + np.abs(fky) ** 2 + np.abs(fkz) ** 2)
    return Pk


def _FFTW_vector_power(f, Lbox, Nsize, fft_object):
    """
  Calculate FFT and power grid before sampling. This function does the main 
  math and physics in power spectrum computation. The fft_object should works on
  one component of the vector field at a time.

  Default normalization is such that
  `np.sum(Pk*(2*np.pi/Lbox)**3)` and `0.5*np.mean(vx**2+vy**2+vz**2)` are equal
  """

    # Fourier transform
    a = (Lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3
    fk = fft_object(f) * a
    # Definition of velocity power spectrum
    Pk = 0.5 * np.sum(np.abs(fk) ** 2, axis=3)
    return Pk


def _scalar_power(f, Lbox, Nsize):
    """
  Calculate FFT and power grid before sampling. This function does the main 
  math and physics in power spectrum computation.

  Default normalization is such that
  `np.sum(Pk*(2*np.pi/Lbox)**3)` and `np.mean(scalar**2)` are equal
  """
    # Fourier transform
    a = (Lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3
    fk = pyfftw.interfaces.numpy_fft.fftn(f) * a
    # Power spectrum
    Pk = 0.5 * np.abs(fk) ** 2
    return Pk


def _FFTW_scalar_power(f, Lbox, Nsize, fft_object):
    """
  Calculate FFT and power grid before sampling. This function does the main 
  math and physics in power spectrum computation.

  Default normalization is such that
  `np.sum(Pk*(2*np.pi/Lbox)**3)` and `np.mean(scalar**2)` are equal
  """
    # Fourier transform
    a = (Lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3
    fk = fft_object(f) * a
    # Power spectrum
    Pk = 0.5 * np.abs(fk) ** 2
    return Pk


def _pair_power(Pk, Lbox, Nsize, shift=np.array([0, 0, 0])):
    """
  Create pairs of power and k. Sampling in concentric spherical
  shells with PkSample to get power spectrum. Called by PowerSpec3D.
  This function is independent of the physics and definition of the
  power spectrum
  """
    # Initialize k space
    Lcell = Lbox / float(Nsize)
    kSpace = 2 * np.pi * np.fft.fftfreq(Nsize, Lcell)
    # Create k and power pairs
    kx, ky, kz = np.meshgrid(kSpace, kSpace, kSpace, indexing="ij")
    # Apply shift
    if shift[0] > 0:
        kx = kx + shift[0]
    if shift[1] > 0:
        ky = ky + shift[1]
    if shift[2] > 0:
        kz = kz + shift[2]
    #
    k = np.sqrt(kx * kx + ky * ky + kz * kz)
    # Construct a (n,2) shape array
    k = np.ravel(k)
    Pk = np.ravel(Pk)
    Pk_pair = np.stack((k, Pk))  # Shape (2,n)
    Pk_pair = np.transpose(Pk_pair)  # Shape (n,2)

    return Pk_pair


def _hist_sample(Pk_pair, kmin, kmax, spacing):
    """ Calculate mean power from Pk pair with specified spacing. """
    bin_centers = np.arange(kmin, kmax + spacing, spacing)
    bin_edges = np.arange(kmin - spacing / 2, kmax + 3 * spacing / 2, spacing)
    Psum, bin_edges_ = np.histogram(
        Pk_pair[:, 0], bins=bin_edges, weights=Pk_pair[:, 1]
    )
    Nsample, bin_edges_ = np.histogram(Pk_pair[:, 0], bins=bin_edges)
    P = Psum / Nsample
    P[Nsample == 0] = 0  # Set P=0 if no sample
    Pvk = np.column_stack((bin_centers, P, Psum, Nsample))

    return Pvk


def high_pass_filter_2d(field, Lbox, low_k=None):  # not very useful
    """
  Remove the k below low_k end of the given image, supposing the low-k end is located at 
  the center of the image. Note that in practice of numpy.fft package, the low-k
  end is at the corner while high-k is in the middle. Should apply 
  np.fft.fftshift() before using this funciton.

  if low_k is not specified, we will take low_k = 2 pi / Lcell.
  """
    dk = 2 * np.pi / Lbox
    Nsize = len(field)
    if low_k is None:
        Lcell = Nsize / Lbox
        low_k = 2 * np.pi / Lcell
    pixel_rad = low_k // dk
    grid = np.arange(0, Nsize)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    x_ctr = x - Nsize // 2
    y_ctr = y - Nsize // 2
    filter = x_ctr ** 2 + y_ctr ** 2 <= pixel_rad ** 2
    field[filter] = 0
    return field




# -----------------------POWER SPECTRUM CLASS--------------------
class PowerSpectrum:
    def __init__(self, Pk, m=0, beta=np.array([-1, -1, -1])) -> None:
        # data
        self.k = Pk[:, 0]
        self.P = Pk[:, 1]
        self.Psum = Pk[:, 2]
        self.Nsample = Pk[:, 3]
        # folding
        self.m = m
        self.beta = beta
        # check alignment
        self.check_alignment()

    def data(self):
        """ Old data stack form. """
        Pk = np.stack([self.k, self.P, self.Psum, self.Nsample], axis=1)
        return Pk

    def subtract_shot_noise(self, Lbox, Np) -> None:
        shot_noise = Lbox ** 3 / Np
        self.P -= shot_noise
        self.P[self.P < 0] = 0

    def __len__(self) -> float:  # len(spctrm) = spctrm.__len__()
        """ Return the number of points of the spectrum, and check if all four
    attributes are aligned while doing so. """
        length = len(self.k)
        if length != len(self.P):
            raise Exception("k and P have different length.")
        elif length != len(self.Psum):
            raise Exception("k and Psum have different length.")
        elif length != len(self.Nsample):
            raise Exception("k and Nsample have different length.")
        else:
            return length

    check_alignment = __len__

    def kmin(self) -> float:
        return np.min(self.k)

    def kmax(self) -> float:
        return np.max(self.k)

    def kres(self) -> float:
        """ The k-space resolution of the power spectrum. """
        kres = (self.kmax() - self.kmin()) / (self.__len__() - 1)
        return kres

    def Lbox(self) -> float:
        Lbox = 2 * np.pi / self.kmin()
        return Lbox

    def energy(self) -> float:
        """ The total energy by direct integral."""
        k = self.k
        dk = k[1:] - k[:-1]
        integral = np.sum(self.P[:-1] * dk)
        return integral

    def copy(self):
        return PowerSpectrum(self.data(), self.m, self.beta)

    def add(self, spctrm) -> None:
        if self.__len__() != len(spctrm):
            raise Exception(
                "Spectra has different length therefore cannot be combined directly."
            )
        else:
            self.Psum += spctrm.Psum
            self.Nsample += spctrm.Nsample
            self.P = self.Psum / self.Nsample * (4 * np.pi * self.k ** 2)

    def remove(self, spctrm) -> None:
        if self.__len__() != len(spctrm):
            raise Exception(
                "Spectra has different length therefore cannot be combined directly."
            )
        else:
            self.Psum -= spctrm.Psum
            self.Nsample -= spctrm.Nsample
            if (self.Nsample < 0).any():
                raise ValueError("Nsample is less than zero.")
            if (self.Psum < 0).any():
                raise ValueError("Psum is less than zero.")
            self.P = self.Psum / self.Nsample * (4 * np.pi * self.k ** 2)

    def append(self, spctrm) -> None:
        kspacing2 = spctrm.kres()
        select = self.k < spctrm.k[0]
        Pk1 = self.data()
        Pk2 = spctrm.data()
        Pk = np.concatenate((Pk1[select], Pk2))
        full_spctrm = PowerSpectrum(Pk)
        # For each k of the appending spectrum
        for k in spctrm.k[spctrm.k < self.k[-1]]:
            # Find where current spectrum is within sampling bins of the appending spectrum.
            select = ((k - kspacing2 / 2) <= self.k) & (self.k < (k + kspacing2 / 2))
            # Then add Psum and Nsample to the wider bins, the appending spectrum
            addPsum = np.sum(self.Psum[select])
            addn = np.sum(self.Nsample[select])
            full_spctrm.Psum[np.where(full_spctrm.k == k)] += addPsum
            full_spctrm.Nsample[np.where(full_spctrm.k == k)] += addn
        select = np.where(full_spctrm.Psum > 0)  # Avoid divided by zero
        # Finally update the power from updated Psum and Nsample
        full_spctrm.P[select] = (
            full_spctrm.Psum[select]
            / full_spctrm.Nsample[select]
            * (4 * np.pi * full_spctrm.k ** 2)
        )
        self = full_spctrm
        self.check_alignment()  # check alignment

    def index(self) -> float:
        # Fit index
        select = self.P > 0
        power, a = np.polyfit(
            np.log10(self.k[select]), np.log10(self.P[select]), 1
        )  # linear fit
        return power

    def peek(self, fit_title=True, remove_zero_power=True) -> None:
        """A quick check on the result. With index shown in the title."""

        plt.figure()
        if fit_title:  # show fit index
            plt.title("$P(k) = k^{%.2f}$" % self.index())
        if remove_zero_power:  # avoid plotting zero values
            select = self.P > 0
            plt.loglog(self.k[select], self.P[select])
        else:
            plt.loglog(self.k, self.P)

        plt.xlabel("$k\,\mathrm{(kpc^{-1})}$")
        plt.ylabel("$P(k)\,\mathrm{(km^2\,s^{-2}\,kpc^{-1})}$")
        plt.grid(True)
        plt.show()

    def plot(
        self, ax=None, remove_zero_power=True, **kwargs
    ) -> plt.Axes:  # copilot wrote this
        """ Plot the power spectrum.
    
    Examples
    --------
    Plot two power spectra for comparison.
    ```
    fig, ax = plt.subplots(figsize=(8,6))
    spctrm1.plot(ax=ax, label='A spectrum')
    spctrm2.plot(ax=ax, label='Another spectrum', linestyle='dashed')
    plt.legend()
    plt.show()
    ```
    """

        if ax is None:  # create a new figure if not specified
            fig, ax = plt.subplots()

        if remove_zero_power:  # avoid plotting zero values
            select = self.P > 0
            ax.loglog(self.k[select], self.P[select], **kwargs)
        else:
            ax.loglog(self.k, self.P, **kwargs)

        ax.set_xlabel("$k\,\mathrm{(kpc^{-1})}$")
        ax.set_ylabel("$P(k)\,\mathrm{(km^2\,s^{-2}\,kpc^{-1})}$")
        ax.grid(True)
        return ax

    def save(self, run_output_dir) -> None:
        """Save the `PowerSpectrum` object under `run_output_dir` using pickle."""
        if (self.beta == np.array([-1, -1, -1])).all():
            filename = os.path.join(run_output_dir, "full_spctrm.pkl")
        else:
            filename = os.path.join(
                run_output_dir, "sub_spctrm_b{}{}{}.pkl".format(*self.beta)
            )
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(run_output_dir, beta=None):
        """Load the saved `PowerSpectrum` object from `run_output_dir` using pickle."""
        if beta is None:
            filename = os.path.join(run_output_dir, "full_spctrm.pkl")
        else:
            filename = os.path.join(
                run_output_dir, "sub_spctrm_b{}{}{}.pkl".format(*beta)
            )
        with open(filename, "rb") as file:
            return pickle.load(file)
# -------------------END POWER SPECTRUM CLASS------------------




# --------------------------SPECTRUM LIST CLASS--------------------
class SpectrumList:
    def __init__(self, spctrm_list):
        self.list = spctrm_list
        self.m = spctrm_list[0].m

    def __len__(self):
        return len(self.list)

    def __getitem__(self, beta) -> PowerSpectrum:
        for spctrm in self.list:
            if (spctrm.beta == beta).all():
                return spctrm
        raise Exception("No spectrum in the list with beta = {}".format(beta))

    def __setitem__(self, beta, spctrm) -> None:
        for i, spctrm in enumerate(self.list):
            if (spctrm.beta == beta).all():
                self.list[i] = spctrm
                return
        self.list.append(spctrm)  # If no match, append to the list
        return

    def __iter__(self):
        return iter(self.list)

    def combine_all(self) -> PowerSpectrum:
        """Construct a power spectrum by combining all spectra in the list."""
        combined = empty_spectrum_like(self.list[0])
        for spctrm in self.list:
            combined.add(spctrm)
        return combined

    def combine_from_beta_sequence(self, beta_sequence=None) -> PowerSpectrum:
        """Construct a power spectrum according to beta sequence."""
        if beta_sequence is None:
            beta_sequence = init_beta_space(m=self.m)
        combined = empty_spectrum_like(self.list[0])
        for beta in beta_sequence:
            combined.add(self[beta])
        return combined

    def append(self, spctrm) -> None:
        self.list.append(spctrm)

    def save(self, run_output_dir) -> None:
        """Save the `SpectrumList` object under `run_output_dir` using pickle."""
        filename = os.path.join(run_output_dir, "spctrm_list.pkl")
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(run_output_dir):
        """
    Load the saved `SpectrumList` object from `run_output_dir`.
    """
        for filename in os.listdir(run_output_dir):
            if filename.startswith("sub_spctrm_b"):
                beta = np.array([int(x) for x in filename[-7:-4]])
                spctrm = PowerSpectrum.load(run_output_dir, beta=beta)
                try:
                    spctrmList.append(spctrm)
                except NameError:
                    spctrmList = SpectrumList([spctrm])  # Create if not exist
        return spctrmList
# --------------------END SPECTRUM LIST-----------------------




def relative_diff(spctrm1, spctrm2, mode="max") -> float:
    """ Return the relative squared difference between two power spectra."""

    if spctrm1.__len__() != spctrm2.__len__():
        raise Exception(
            "Spectra has different length therefore cannot be compared directly."
        )
    else:
        P1 = spctrm1.P
        P1[np.isnan(P1)] = 0

        P1[P1 == 0] = 1e-10  # avoid divided by zero
        print(
            "Divided by zero encountered in (P1-P2)/P1. Avoided by setting P1 = 1e-10."
        )

        P2 = spctrm2.P
        P2[np.isnan(P2)] = 0
        if mode == "mean":
            return np.mean(((P1 - P2) / P1) ** 2) ** 0.5
        elif mode == "max":
            return np.max(abs(P1 - P2) / P1)
        elif mode == "sum":
            return np.sum(((P1 - P2) / P1) ** 2) ** 0.5
        else:
            raise Exception("Mode not recognized. Use 'mean' or 'max'.")


def empty_spectrum_like(spctrm, keep_m=False, keep_beta=False) -> PowerSpectrum:
    """ Return an zero power spectrum object with the same k, m, beta as the input. """
    k = spctrm.k
    zeros = np.zeros_like(k)
    Pvk = np.column_stack((k, zeros, zeros, zeros))
    m = spctrm.m if keep_m else 0
    beta = spctrm.beta if keep_beta else np.array([-1, -1, -1])
    return PowerSpectrum(Pvk, m=m, beta=beta)


def load_spectrum(filename) -> PowerSpectrum:
    """ Read a power spectrum from a file using Pickle. """
    with open(filename, "rb") as f:
        spctrm = pickle.load(f)
    return spctrm


def init_beta_space(m):
    """A default space of beta of shape (m^3, 3)."""
    bi = np.arange(0, m)
    bj = np.arange(0, m)
    bk = np.arange(0, m)
    beta_space = np.array(np.meshgrid(bi, bj, bk, indexing="ij")).T.reshape(-1, 3)
    return beta_space


def random_beta_sequence(m, seed=1):
    """Create a random beta sequence of shape (m^3, 3)."""
    np.random.seed(seed)
    beta_space = init_beta_space(m)
    np.random.permutation(beta_space)
    return beta_space
