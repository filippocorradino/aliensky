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


class Atmosphere:

    Border = Enum('Border', 'GND TOA')

    def __init__(self):
        # Model Params
        self.Rt = 6420000
        self.Rb = 6360000
        self.Hr = 8000
        self.Hm = 1200
        # Calcs Params
        self.Nf = 3
        self.Nh = 60
        self.Nm = 36
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
        self.Lstar = np.array([255, 255, 255])  # TODO: use actual values
        self.ARstar = np.deg2rad(5/2)  # Angular radius of the star
        self.AAstar = np.pi*(self.ARstar**2)  # Angular area of star
        self.cosARstar = np.cos(self.ARstar)  # Cosine of angular radius of star
        self.albedo = 0.39
        # Precalculate
        self._precalculate()
        # Checks
        assert len(self.lambdas) == self.Nf
        assert len(self.beta0_s_r) == self.Nf
        assert len(self.beta0_s_m) == self.Nf
        assert len(self.Lstar) == self.Nf

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
        # x,v --> r,m --> ur,um

    def total_radiance(self, x, v, s, inscatter=True):
        """
        x is the local position vector in planet-frame [m]
        v is the view direction versor in planet-frame
        s is the star direction versor in planet-frame
        """
        r, m = self._xv2rm(x, v)
        t0, endray = self._endray(r, m)
        x0 = x + t0*v
        n0 = x0 / np.linalg.norm(x0)
        Txv = self._transmittance(x, v)
        # 1 - Direct light
        L0 = 0
        if endray is Atmosphere.Border.TOA:
            if np.dot(s, v) > self.cosARstar:
                # TODO: scale radiance if only part of the pixel is occupied?
                L0 = np.multiply(self.Lstar, Txv)
        # 2 - Reflected light
        # NB: no inscatter considered in sun-ground ray
        R = 0
        if endray is Atmosphere.Border.GND:
            I = self.albedo * self.AAstar * \
                np.multiply(np.dot(s, n0),
                            np.multiply(self.Lstar,
                                        self._transmittance(x0, s)))
            R = np.multiply(I, Txv)
        # 3 - Inscatter
        # TODO: add inscatter from ground reflected light!
        S = np.zeros(self.Nf)
        if inscatter:
            t_int = t0*self.Vint
            y_int = x + np.outer(t_int, v)
            h_int = np.linalg.norm(y_int, axis=1) - self.Rb
            mu = np.dot(v, s)
            Pr = self._Pr(mu)
            Pm = self._Pm(mu)
            beta_s_r = np.outer(np.exp(-h_int/self.Hr), self.beta0_s_r)
            beta_s_m = np.outer(np.exp(-h_int/self.Hm), self.beta0_s_m)
            f_int = np.zeros((self.Nint, self.Nf))
            for iy, y in enumerate(y_int):
                # Txy * J[L](x,v,s) = Txy * sum(beta_s_i*Pi(v.s)*L(y,s,s)*As)
                Lyss = self.total_radiance(y, s, s, inscatter=False)
                f_int[iy] = \
                    np.multiply(np.divide(Txv, self._transmittance(y, v)),
                                np.multiply(Lyss,
                                            Pr*beta_s_r[iy, :] +
                                            Pm*beta_s_m[iy, :]))
            f_int = f_int * self.AAstar
            for il, l in enumerate(self.lambdas):
                S[il] = np.trapz(f_int[:, il], t_int)
        # Totals
        L = L0 + R + S
        return L

    # def _Pr(self, mu):
    #     return 3/16/np.pi*(1+mu**2)

    # def _Pm(self, mu):
    #     g = self.g_mie
    #     return 3/8/np.pi*((1-g**2)*(1+mu**2))/((2+g**2)*((1+g**2-2*g*mu)**1.5))

    def _xv2rm(self, x, v):
        """
        x is the local position vector in planet-frame [m]
        v is the view direction versor in planet-frame
        r is the local radial distance from centre of planet [m]
        m is the cosine of the view zenith angle
        r = |x|
        m = (x.v/r)
        """
        r = np.linalg.norm(x)
        m = np.dot(x, v) / r
        return r, m

    def _endray(self, r, m):
        """
        Evaluates the length of a view ray and whether it ends on the ground or
        at the top of the atmosphere
        r is the local radial distance from centre of planet [m]
        z is the view zenith angle [rad]
        m is the cosine of the view zenith angle (m = cos(z))
        view ray parametric coordinates: [r+cos(z)t, sin(z)t]
        t at intersections with a shell of radius R:
        t = -r*m +/- r*sqrt((R/r)^2 + m^2 - 1)
        """
        delta_b = (self.Rb/r)**2 + m**2 - 1
        delta_t = (self.Rt/r)**2 + m**2 - 1
        if (m < 0) and (delta_b > 0):
            # End of ray is on the ground
            # Take first intersection - second one is through Earth
            t0 = r * (-m - np.sqrt(delta_b))
            return t0, Atmosphere.Border.GND
        else:
            # End of ray is on the top of the atmosphere
            # Take second intersection - first one is behind us
            t0 = r * (-m + np.sqrt(delta_t))
            return t0, Atmosphere.Border.TOA

    def _transmittance(self, x, v):
        """
        x is the local position vector in planet-frame [m]
        v is the view direction versor in planet-frame
        r is the local radial distance from centre of planet [m]
        z is the view zenith angle [rad]
        m is the cosine of the view zenith angle (m = cos(z))
        n is the   sine of the view zenith angle (n = sin(z))
        view ray parametric coordinates: [r+cos(z)t, sin(z)t]
        t at intersections with a shell of radius R:
        t = -r*m +/- r*sqrt((R/r)^2 + m^2 - 1)
        """
        r, m = self._xv2rm(x, v)
        n = np.sqrt(1 - m**2)  # Limits z to [0, pi] without loss of generality
        t0, _ = self._endray(r, m)
        t_int = t0*self.Vint
        y_int = np.array([r+t_int*m, t_int*n])
        h_int = np.linalg.norm(y_int, axis=0) - self.Rb
        T_vec = np.zeros(self.Nf)
        for il, l in enumerate(self.lambdas):
            f_int = (self.beta0_e_r[il] * np.exp(-h_int/self.Hr) +
                     self.beta0_e_m[il] * np.exp(-h_int/self.Hm))
            T_vec[il] = np.exp(-np.trapz(f_int, t_int))
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
    s = np.array([1, 0, 0])
    s = s / np.linalg.norm(s)
    image_size = 32
    radius = image_size/2
    rgbArray = np.zeros((image_size, image_size, 3), 'uint8')
    counter = 0
    p_vec = np.linspace(-radius, +radius, image_size)
    for ix in range(image_size):
        for iy in range(image_size):
            px = p_vec[ix]
            py = p_vec[iy]
            el = np.pi/2 * (1 - np.sqrt(px**2 + py**2)/radius)
            if el > 0:
                az = np.arctan2(py, px)
                v = np.array([np.cos(el)*np.cos(az),
                              np.cos(el)*np.sin(az),
                              np.sin(el)])
                radiance = atmosphere.total_radiance(x, v, s) * 1000
                rgbArray[ix, iy, :] = radiance
                print("{:08d} of {:08d} pixels rendered - radiance: {}"
                      .format(counter, image_size**2, radiance))
            counter = counter + 1
    img = Image.fromarray(rgbArray)
    img.save('myimg.jpeg')
    return


if __name__ == "__main__":
    main()