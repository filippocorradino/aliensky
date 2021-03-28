#!/usr/bin/env python
# encoding: utf-8
"""
Alien Sky

[BR08] E. Bruneton, F. Neyret "Precomputed Atmospheric Scattering"
       Proceedings of the 19th Eurographics conference on Rendering, 2008
       https://doi.org/10.1111/j.1467-8659.2008.01245.x
       https://hal.inria.fr/inria-00288758/document
"""

__author__ = "Filippo Corradino"
__email__ = "filippo.corradino@gmail.com"

from enum import Enum
from PIL import Image
from scipy.interpolate import interp1d, RectBivariateSpline
import numpy as np

import cProfile
import io
import pstats

# TODO: switch from n to N as vectors lengths


class Atmosphere:

    Border = Enum('Border', 'GND TOA')

    def __init__(self):
        # Model Params
        self.Rt = 6420000
        self.Rb = 6360000
        self.Hr = 8000
        self.Hm = 1200
        # Calcs Params
        self.Nl = 3
        self.Nr = 60  # Altitude layers
        self.Nm = 36  # Zenith angle layers
        self.Nint = 50
        self.Vint = np.linspace(0, 1, self.Nint)
        # Initialization
        self.lambdas = np.array([680e-9, 550e-9, 440e-9])
        self.beta0_s_r = np.array([5.8e-6, 13.5e-6, 33.1e-6])
        self.beta0_s_m = np.array([1e-6, 1e-6, 1e-6])  # TODO: use actual values
        self.beta0_e_r = 1.0*self.beta0_s_r
        self.beta0_e_m = 2.1*self.beta0_s_m
        self.g_mie = 0.76
        # Star/Planet params  # TODO: move them to star/planet definition
        self.Lstar = np.array([[255, 255, 255]]).T  # TODO: use actual values - Needs to be 2D for ease
        self.ARstar = np.deg2rad(5/2)  # Angular radius of the star
        self.AAstar = np.pi*(self.ARstar**2)  # Angular area of star
        self.cosARstar = np.cos(self.ARstar)  # Cosine of angular radius of star
        self.albedo = 0.39
        # Precalculate
        self._precalculate()
        # Checks
        assert len(self.lambdas) == self.Nl
        assert len(self.beta0_s_r) == self.Nl
        assert len(self.beta0_s_m) == self.Nl
        assert len(self.Lstar) == self.Nl

    def _precalculate(self):
        # Scattering Coefficients
        g = self.g_mie
        m_samples = np.linspace(-1, +1, self.Nm)
        Pr_samples = 3/16/np.pi * (1+m_samples**2)
        Pm_samples = 3/8/np.pi * (((1-g**2)*(1+m_samples**2)) /
                                  ((2+g**2)*((1+g**2-2*g*m_samples)**1.5)))
        self._Pr = interp1d(m_samples, Pr_samples)
        self._Pm = interp1d(m_samples, Pm_samples)
        # Transmittance
        ur_samples = np.linspace(0, 1, self.Nr)  # TODO: make into logspace
        um_samples = np.linspace(0, 1, self.Nm)
        # TODO: vectorize
        self._T = []
        count = 0
        total = self.Nr*self.Nm
        Tmap = np.zeros((self.Nl, self.Nr, self.Nm))
        for ir in range(self.Nr):
            for im in range(self.Nm):
                Tmap[:, ir, im] = self._transmittance_precalc(
                    ur_samples[ir], um_samples[im])
                count += 1
                print(f"Processed {count} of {total}: {Tmap[:, ir, im]}")
        for il in range(self.Nl):
            Tcur = np.squeeze(Tmap[il, :, :])
            self._T.append(RectBivariateSpline(ur_samples, um_samples, Tcur))
        # x,v --> r,m --> ur,um

    def total_radiance(self, x, v, s, inscatter=True):
        """Total spectral radiance computation

        Args:
            x: (3,) or (3, n) np.ndarray
               the local position vector in planet-frame [m]
            v: (3,) or (3, n) np.ndarray
               the view direction versor in planet-frame
            s: (3,) or (3, n) np.ndarray
               the star direction versor in planet-frame
            inscatter: bool
               whether to consider inscattered light

        Returns:
            L: (Nl,) or (Nl, n) np.ndarray
               spectral radiance along the view versor
        """
        # Reshape all inputs to 2D (3, n) vectors
        ins = (x, v, s)
        for vec in ins:
            vec.shape = (3, -1)
        n = max(x.shape[1] for x in ins)
        # FIXME: find a way to loop over this:
        if x.shape[1] == 1 and n > 1:
            x = np.dot(x, np.ones((1, n)))
        if v.shape[1] == 1 and n > 1:
            v = np.dot(v, np.ones((1, n)))
        if s.shape[1] == 1 and n > 1:
            s = np.dot(s, np.ones((1, n)))
        # Calculations
        r, m = self._xv2rm(x, v)
        t0, endray = self._endray(r, m)
        x0 = x + t0*v
        n0 = x0 / np.linalg.norm(x0, axis=0)
        Txv = self._transmittance(x, v).reshape((self.Nl, n))
        L0 = np.zeros((self.Nl, n))
        S = np.zeros((self.Nl, n))
        R = np.zeros((self.Nl, n))
        ig = (endray == self.Border.GND)   # GND intersect logical vector
        ix = (np.sum(s*v, axis=0) > self.cosARstar)
        # 1 - Direct light
        # TODO: scale radiance if only part of the pixel is occupied?
        L0[:, ix] = self.Lstar * Txv[:, ix]
        # 2 - Reflected light
        # NB: no inscatter considered in sun-ground ray
        I = self.albedo * self.AAstar * np.sum(s*n0, axis=0, keepdims=True)
        I = np.dot(self.Lstar, I) * self._transmittance(x0, s)
        R[:, ig] = I[:, ig] * Txv[:, ig]
        # 3 - Inscatter
        # TODO: add inscatter from ground reflected light!
        if inscatter:
            # Create a third axis along which to integrate
            # f_int (Nl, n, Nint)
            # t_int (3, n, Nint)
            # y_int (3, n, Nint)
            f_int = np.zeros((self.Nl, n, self.Nint))
            t_int = np.tile(np.outer(t0, self.Vint), (3, 1, 1))
            y_int = t_int * v.reshape((3, n, 1))
            y_int = y_int + x.reshape((3, n, 1))
            h_int = np.linalg.norm(y_int, axis=0) - self.Rb  # (n, Nint)
            mu = np.sum(v*s, axis=0)
            Pr = self._Pr(mu).reshape(1, n)
            Pm = self._Pm(mu).reshape(1, n)
            beta_s_r = np.tile(np.exp(-h_int/self.Hr), (self.Nl, 1, 1)) * \
                np.reshape(self.beta0_s_r, (self.Nl, 1, 1))
            beta_s_m = np.tile(np.exp(-h_int/self.Hm), (self.Nl, 1, 1)) * \
                np.reshape(self.beta0_s_m, (self.Nl, 1, 1))
            # Txy * J[L](x,v,s) = Txv * sum(beta_s_i*Pi(v.s)*L(y,s,s)*As)
            # TODO: vectorize this as well?
            for iy in range(self.Nint):
                # Txy * J[L](x,v,s) = Txv * sum(beta_s_i*Pi(v.s)*L(y,s,s)*As)
                y = y_int[:, :, iy]
                Lyss = self.total_radiance(y, s, s, inscatter=False)
                f_int[:, :, iy] = Txv / self._transmittance(y, v) * Lyss * \
                    (Pr*beta_s_r[:, :, iy] + Pm*beta_s_m[:, :, iy])
                print(f"Integration step: {iy+1} of {self.Nint}")
            f_int = f_int * self.AAstar
            S = np.trapz(f_int, t_int, axis=2)
        # Totals
        L = L0 + R + S
        return L

    @staticmethod
    def _xv2rm(x, v):
        """(x, v) to (r, m) coordinate transformation

        Args:
            x: (3,) or (3, n) np.ndarray
               the local position vector in planet-frame [m]
            v: (3,) or (3, n) np.ndarray
               the view direction versor in planet-frame

        Returns:
            r: float or (n,) np.ndarray
               the local radial distance from centre of planet [m]
            m: float or (n,) np.ndarray
               the cosine of the view zenith angle (1 = up, -1 = down)
        """
        r = np.linalg.norm(x, axis=0)
        m = np.sum(x*v, axis=0) / r
        return r, m

    def _urum2rm(self, ur, um):
        """(ur, um) to (r, m) coordinate transformation

        Args:
            ur: float or (n,) np.ndarray
                the adimensional height in the atmosphere [0, 1]
            um: float or (n,) np.ndarray
                the adimensional zenith angle [0, 1]

        Returns:
            r: float or (n,) np.ndarray
               the local radial distance from centre of planet [m]
            m: float or (n,) np.ndarray
               the cosine of the view zenith angle (1 = up, -1 = down)
        """
        # TODO: make a shader?
        # mu = +1  um = 1.0  (Straight up)
        # mu = muh um = 0.5  (Horizon-grazing)
        # mu = -1  um = 0.0  (Straight down)
        r = self.Rb + (self.Rt-self.Rb) * ur
        muh = -np.sqrt(1 - (self.Rb/r)**2)
        ig = (um < 0.5)  # GND intersect logical vector
        # GND intersect: m = 2*(muh+1)*um - 1       = +2*muh*um +2*um -1
        # TOA intersect: m = (1-muh)*(2*um-1) + muh = -2*mum*um +2*um -1 +2*muh
        m = ((1-muh)*(2*um-1) + muh)*(1-ig) + (2*(muh+1)*um - 1)*ig
        return r, m

    def _rm2urum(self, r, m):
        """(r, m) to (ur, um) coordinate transformation

        Args:
            r: float or (n,) np.ndarray
               the local radial distance from centre of planet [m]
            m: float or (n,) np.ndarray
               the cosine of the view zenith angle (1 = up, -1 = down)

        Returns:
            ur: float or (n,) np.ndarray
                the adimensional height in the atmosphere [0, 1]
            um: float or (n,) np.ndarray
                the adimensional zenith angle [0, 1]
        """
        # TODO: make a shader?
        ur = (r-self.Rb) / (self.Rt-self.Rb)
        muh = -np.sqrt(1 - (self.Rb/r)**2)  # TODO: map?
        t0, boundary = self._endray(r, m)
        ig = (boundary == self.Border.GND)  # GND intersect logical vector
        um = ((m+1)/(muh+1)/2) * ig + (((m-muh)/(1-muh)+1)/2) * (1-ig)
        return ur, um

    def _endray(self, r, m):
        """View ray propagation end

        Args:
            r: float or (n,) np.ndarray
               the local radial distance from centre of planet [m]
            m: float or (n,) np.ndarray
               the cosine of the view zenith angle (1 = up, -1 = down)

        Returns:
            t0: float or (n,) np.ndarray
                the distance to the view ray end
            border: Border enum or (n,) np.ndarray of Border enums
                the border at which the view ray ends, whether ground or the
                top of the atmosphere
        """
        r = np.asarray(r).reshape(-1)  # Make a (n,) array
        m = np.asarray(m).reshape(-1)  # Make a (n,) array
        delta_b = (self.Rb/r)**2 + m**2 - 1
        delta_t = (self.Rt/r)**2 + m**2 - 1
        ig = np.logical_and(m < 0, delta_b > 0)  # GND intersect logical vector
        # GND intersect: Take first intersection - second one is through Earth
        # TOA intersect: Take second intersection - first one is behind us
        border = np.array([[self.Border.TOA, self.Border.GND][x] for x in ig*1])  # HACK
        if len(border) == 1:
            border = border.item()
        t0 = r * (-m - np.sqrt(delta_b*ig) + np.sqrt(delta_t*(1-ig)))
        return t0, border

    def _transmittance_precalc(self, ur, um):
        """Transmittance along the view vector, calculated on adimensional mesh

        Args:
            ur: float
                the adimensional height in the atmosphere [0, 1]
            um: float
                the adimensional zenith angle [0, 1]

        Returns:
            T_list: list of floats
                    the list of spectral transmittances along the view vector
        """
        # TODO: vectorize
        # view ray parametric coordinates: [r+mt, nt]
        r, m = self._urum2rm(ur, um)
        n = np.sqrt(1 - m**2)  # Limits z to [0, pi] without loss of generality
        t0, _ = self._endray(r, m)
        t_int = t0*self.Vint
        y_int = np.array([r+t_int*m, t_int*n])
        h_int = np.linalg.norm(y_int, axis=0) - self.Rb
        T_list = np.zeros(self.Nl)
        for il, l in enumerate(self.lambdas):
            f_int = (self.beta0_e_r[il] * np.exp(-h_int/self.Hr) +
                     self.beta0_e_m[il] * np.exp(-h_int/self.Hm))
            T_list[il] = np.exp(-np.trapz(f_int, t_int))
        return T_list

    def _transmittance(self, x, v):
        """Transmittance along the view vector

        Args:
            x: (3,) or (3, n) np.ndarray
               the local position vector in planet-frame [m]
            v: (3,) or (3, n) np.ndarray
               the view direction versor in planet-frame

        Returns:
            T_vec: (Nl,) or (Nl, n) np.ndarrays
                   the spectral transmittances along the view vector
        """
        r, m = self._xv2rm(x, v)
        ur, um = self._rm2urum(r, m)
        T_vec = np.array([T.ev(ur, um) for T in self._T])
        return T_vec


def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE  # 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    return wrapper


@profile
def main():
    atmosphere = Atmosphere()
    x = np.array([0, 0, atmosphere.Rb])
    s = np.array([1, 0, 0.5])
    s = s / np.linalg.norm(s)
    image_size = 256
    radius = image_size/2
    p_vec = np.linspace(-radius, +radius, image_size)
    px_mesh, py_mesh = np.meshgrid(p_vec, p_vec)
    el_mesh = np.pi/2 * (1 - np.sqrt(px_mesh**2 + py_mesh**2)/radius)
    az_mesh = np.arctan2(py_mesh, px_mesh)
    el = el_mesh.reshape((-1,))
    az = az_mesh.reshape((-1,))
    v = np.stack((np.cos(el)*np.cos(az),
                  np.cos(el)*np.sin(az),
                  np.sin(el)))
    radiance = atmosphere.total_radiance(x, v, s) * 1000
    rgbArray = radiance.T.reshape((image_size, image_size, 3)).astype('uint8')
    rgbArray[(el_mesh < 0), :] = 0  # Mask elevation < 0
    img = Image.fromarray(rgbArray)
    img.save('myimg.jpeg')
    return


if __name__ == "__main__":
    main()
