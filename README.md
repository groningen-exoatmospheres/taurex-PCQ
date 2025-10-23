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

## As an extension of TauREx-PyMieScatt, this pulgin includes the same  capabilities

- ✅ **Supports multiple particle size distributions:**
  - `normal` (log-normal)
  - `budaj` (2015)
  - `deirmendjian` (1964)
- ✅ **Multiple species and per-species fitting**
- ✅ **Agregates** Can compute and retrieve whatever combinason of specie as a single opacity source. Example: You can use an agreagate of SiO2 + MgSiO3 + Mg3SiO4 as one specie for your retrieval / model. Aggregate theory is from Akimasa et al. 2014.
- ✅ **Particle decay with altitude** (`exp_decay` based on Whitten 2008 / Atreya 2005)
- ✅ **Computes exctinction** using the species optical constant via **Effective Medium Theory (Bohren & Huffman 1983)**
- ✅ **Multiple fittable parameters** for TauREx retrievals

---

## 🔧 Model Parameters

| Name | Description |
|------|-------------|
| `species` | Your name for the species included through the `mie_species_path` parameter. This name will be used as suffixes added to the other parameters to distinguish between included species. |
| `mie_species_path` | Paths to the `Q_ext` grids of the aerosols you want to include |
| `mie_particle_radius_distribution` | `"normal"`, `"budaj"`, or `"deirmendjian"` |
| `mie_particle_mean_radius` | Mean particle radius (µm) |
| `mie_particle_logstd_radius` | Log-normal std dev (for `"normal"` distribution) |
| `mie_particle_paramA/B/C/D` | Parameters for Deirmendjian distribution |
| `mie_particle_mix_ratio` | Number density (molecules/m³) |
| `mie_midP` | Pressure at cloud center (Pa) |
| `mie_rangeP` | Extend of the clouds in log scale around `mie_midP`. If `mie_midP` = 1e5 Pa and `mie_rangeP` = 1 then clouds extend from 1e6 to 1e4 Pa |
| `mie_particle_altitude_distrib` | Currently supports `'exp_decay'` or `'linear'` |
| `mie_particle_altitude_decay` | Decay exponent per species e.g `-5` |

---

## 💡 Usage Example in a TauREx parameter file or 'parfile'

```python
[Model]
model_type = transit

    [[PyMieScattGridExtinction]]

    species = SiO, Mg2SiO4_glass, custom_molecule
    mie_species_path = path_to_SiO.h5 , path_to_Mg2SiO4_glass.h5 , path_to_custom_molecule.h5   #e.g. You can use the optical constant from Kitzmann and Heng 2018
    mie_particle_radius_distribution = budaj
    mie_particle_mean_radius = 0.1, 0.4 , 10
    mie_midP = 1e5, 1e2, 1
    mie_rangeP = 3 , 1 , 2
    mie_particle_mix_ratio = 1e5, 1e8 , 10e3
    mie_particle_radius_Nsampling = 5
    mie_particle_altitude_distrib = linear
```
---
## Limitations

As the `Q_ext` are computed for a range of radii, it is strongly recomended that the `mie_particle_mean_radius` prior does not extend beyond this range. Any radius outside of the range will use the `Q_ext` of the closest radius present in the grid.
