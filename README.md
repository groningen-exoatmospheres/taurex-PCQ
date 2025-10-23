# Faster clouds for TauREx 3

This plugin provides **a new nethod to inclue aerosols** in [TauREx 3](https://github.com/ucl-exoplanets/TauREx3_public), extending the TauREx-PyMieScatt plugin.  
It speeds up the inclusion of physically consistent cloud and aerosol opacity modeling by using precomputed extinction efficiency (`Q_ext`) grids generated with [PyMieScatt](https://pymiescatt.readthedocs.io/en/latest/).

---

A list of precomputed `Q_ext` grids for molecules such as Silicates or Titan Tholin aerosols are available at: (repo)

---

## 🔧 Features

- ✅ Compatible with `transit` and `emmsion` models.
- ✅ Works with any aerosol specie given that the user provides a `.h5` file with :

- A `radius_grid` dataset with the particule sizes in microns ( length `a` ).
- A `wavenumber_grid` dataset with the wavenumber at which the `Q_ext` were computed in cm-1 ( length `b` )
- A `Qext_grid` dataset with the computed `Q_ext` from PyMieScat ( length (`a`, `b`) )
---

