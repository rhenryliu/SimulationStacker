# SIMBA `ElectronAbundance`: two normalisations in one field

Investigation record, 2026-09-22/23, updated 2026-10-14 (section 11). Read-only
checks of the SIMBA and CAMELS snapshots on scratch, of the public GIZMO/Grackle
sources and documentation, and of published work that uses SIMBA. No code or
cached field was changed; the only data written were the four approved CAMELS
downloads (section 5).

**Status 2026-10-14:** the SIMBA team has tentatively confirmed the bug. They
recommend a correction built from the Grackle species alone. It agrees with ours
to 1e-6 in every global budget we checked, and to 1e-4 per density bin except
one bin set by a single particle (section 11).

Short version: for roughly 70% of SIMBA gas particles (and 72-79% of CAMELS-SIMBA
gas particles) the snapshot field `PartType0/ElectronAbundance` is **not**
n_e/n_H, the convention documented in the GIZMO user guide and assumed by every
tool and paper we found. For these particles it is Grackle's internal
normalisation n_e m_p / rho = X_H n_e/n_H. The Grackle species fields of each
particle use the same normalisation as its EA, so the hydrogen sum
GrackleHI + GrackleHII (X_H(1-Z) or 1) identifies the class. Dust tracks the
class closely but not exactly (section 3.3). The correct n_e/n_H is recovered,
for all but ~3e-7 of the gas mass (section 11.3), by the SIMBA team's
species-ratio formula (recommended),

```
n_e/n_H = GrackleHII / (GrackleHI + GrackleHII)
        + y (GrackleHeII + 2 GrackleHeIII) / (GrackleHeI + GrackleHeII + GrackleHeIII)
y = (1 - X_H) / (4 X_H) = 0.0789   (X_H = 0.76; maximum 1 + 2y = 1.1579)
```

or, equivalently wherever EA and the species share a normalisation, by our
original form

```
n_e/n_H = min( ElectronAbundance / (GrackleHI + GrackleHII) , 1 + 2y )
```

The two corrections disagree only for jet-launched particles, ~3e-7 of the gas
mass (section 11.3). That is a different set from the ~2e-7 of the mass above
the cap in section 3.8, although the jet particles appear in both.

Reading EA as n_e/n_H undercounts SIMBA's electrons by ~17% (tau, kSZ, DM) and
biases temperatures inferred from it high by 14% in the affected ionized gas.

---

## 1. The two candidate meanings

| Reading | Definition | Fully ionized primordial gas | H ionized, He singly ionized |
|---|---|---|---|
| Documented (GIZMO/Gadget/TNG) | n_e / n_H | 1 + 2y = **1.158** | 1 + y = 1.079 |
| Grackle internal | n_e m_p / rho = X_H n_e/n_H | X + Y/2 = **0.880** | X + Y/4 = 0.820 |

`src/mapMaker.py` uses the documented reading for TNG and SIMBA alike
(`make_mass_field`, ionized branch: `Ne = xe * X_H * Mgas_g / m_p`;
`make_sz_field`: `ne = EA*X_H*D/m_p`, `Te = (gamma-1) IE/k_B * 4 m_p/(1+3X_H+4X_H EA)`).

## 2. Primary sources: code and documentation

All retrieved to stdout on 2026-09-22/23; line numbers refer to the stated
revision.

**GIZMO (public repository, bitbucket.org/phopkins/gizmo-public).** SIMBA ran a
private fork of GIZMO; the oldest public commit (6096d8c, 2019-08) postdates the
SIMBA runs.

- `io.c` (master): `case IO_NE` writes `SphP[pindex].Ne` as `"ElectronAbundance"`;
  `case IO_NH` writes `SphP[pindex].grHI` when `COOL_GRACKLE_CHEMISTRY > 0`
  (lines 311-319). So `NeutralHydrogenAbundance` is the Grackle HI field.
- `cooling/grackle.c` (master, lines 66-74): `ne_density = density * ne_guess;`
  and `HI_density = density * SphP[target].grHI; //initialized with HYDROGEN_MASSFRAC`.
  After `solve_chemistry` the species are written back as `HI_density / density`
  etc. (lines 111-117). In the public code the updated electron field is *not*
  written back to `Ne` in this branch.
- `init.c` (master, lines 595, 862-867): `SphP[i].Ne = 1.0;`,
  `grHI = HYDROGEN_MASSFRAC`, `grHeI = 1.0 - HYDROGEN_MASSFRAC`, i.e. the Grackle
  species are **mass fractions**.
- `cooling/cooling.c` (6096d8c, lines 602-603, 623): GIZMO's own equilibrium
  solver sets `n_elec = nHp + nHep + 2 * nHepp;`, `necgs = n_elec * nHcgs;` and
  `SphP[target].Ne = n_elec;`, i.e. **n_e/n_H**, with H and He abundances per H
  nucleus (He total `yhelium` = Y/4X).
- **GIZMO user guide** (tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html):
  "ElectronAbundance ... this is the mean free-electron number per proton
  (hydrogen nucleon); averaged over the mass of the gas particle", and
  "NeutralHydrogenAbundance ... neutral hydrogen fraction (between 0 and 1)".

**A public fork of the pre-2019 GIZMO lineage** (github.com/egentry/gizmo-clustered-SNe,
master): `cooling/grackle.c` line 112 `my_fields.e_density[0] = rho * *ne_guess;`,
line 166 `*ne_guess = my_fields.e_density[0] / my_fields.density[0];`, called on
`SphP[i].Ne` (`cooling/cooling.c` lines 62, 97). In this lineage `Ne` therefore
*stores* Grackle's `e_density / rho`.

**Grackle 3.1** (github.com/grackle-project/grackle, tag grackle-3.1):
`doc/source/Integration.rst` lines 391-398: "the electron mass density should be
scaled by the ratio of the proton mass to the electron mass such that the
electron density in the code is the electron number density times the
**proton** mass." `src/clib/calculate_temperature.c` lines 140-142:
`number_density = 0.25*(HeI+HeII+HeIII) + HI + HII + e_density`, confirming the
same normalisation. Hence `e_density / density = n_e m_p / rho = X_H n_e/n_H`.

**Grackle self-shielding option 3** (`doc/source/Parameters.rst`, grackle-3.1,
lines 275-282): "Approximate self-shielding in both HI and HeI, but ignoring
HeII ionization and heating from the UV background entirely (HeII ionization and
heating rates are set to zero)." Public GIZMO master sets
`self_shielding_method=3`; the 2019 public version does not set it. Davé et al.
2019 (Sec. 2.1) states self-shielding "based on the Rahmati et al. 2013
prescription" with "Haardt & Madau 2012, modified to account for self-shielding
(A. Emerick, priv. comm.)" but not the option number.

**SIMBA team's SWIFT port** (github.com/romeeld/swiftsim, master; attributed to
R. Davé by the literature search, account owner not independently confirmed):
- `src/cooling/SIMBA/cooling_struct.h` line 45: "here all fractions are mass fraction";
- `src/cooling/SIMBA/cooling.c` lines 137-139: `e_frac = HII_frac + 0.25*HeII_frac + 0.5*HeIII_frac`;
  line 364: `e_frac = *data->e_density / rho;`
- `src/cooling/KIARA/cooling.c` line 636: `e_frac = *data->e_density * rhoinv;`,
  passed to `cooling_convert_u_to_temp`, whose docstring (`KIARA/cooling.h`
  line 278) reads "@param ne Electron number density relative to H atom density".

This is SWIFT, not the GIZMO fork that produced the SIMBA snapshots, but it is
SIMBA-team code that stores the Grackle-normalised quantity and hands it to a
function documented as taking n_e/n_H: the same mix-up that explains the data.

## 3. Data tests

Snapshots: SIMBA m100n1024 s50 (snapshot 125, z=0.49, 1.03e9 gas particles;
snapshot 151, z=0), m50n512 {s50, s50nox, s50nojet, s50noagn, s50nofb}
(snapshot 125), all under `/pscratch/sd/r/rhliu/simulations/SIMBA/`. X_H = 0.76.

### 3.1 Species normalisation splits the gas into two classes, with nothing in between

- **Mass-fraction class**: GrackleHI+HII+HM = 0.76(1-Z), He total = 0.24(1-Z)
  (so H + He + Z = 1, and n_He/n_H = y exactly; checked to 1e-7 on 5e6
  m50n512 s50 particles, 2026-10-14; the original text said He = 0.24).
- **Per-H class**: GrackleHI+HII+HM = 1 exactly, He total = 0.07895 = Y/4X.

GrackleHM is negligible (HM/H <= 1.5e-20 in the same m50n512 subsample). In
every snapshot tested, 0 particles fall outside these two classes (re-confirmed
on the full m50n512 s50 and CAMELS-SIMBA CV_0 snapshots, section 11).

### 3.2 EA matches its own particle's species in that particle's convention

| Snapshot | Mass-frac class (particles) | EA = HII + HeII/4 + HeIII/2 (mass-frac class) | EA = HII + HeII + 2 HeIII (per-H class) |
|---|---|---|---|
| m100n1024 z=0.49 (5e6 random) | 69.3% | 100.000% | 97.15% |
| m100n1024 z=0 (5e6 random) | 62.7% | 100.000% | 98.29% |
| m50n512 s50 z=0.49 (all 1.28e8) | 67.9% | 100.000% | 97.41% |

(Tolerance 0.2%.) The per-H column counts nearly neutral particles, where a
relative tolerance is ill-defined, as mismatches. Excluding particles with
species-derived n_e/n_H <= 0.02 (2.9% of the per-H class in m50n512 s50), 99.45%
match at 0.2% (5e6-particle m50n512 subsample) and all but 15 of 4.0e7 match at
1% (full m50n512). The 0.2-1% residual (0.17% of all particles) is dense
(n_H ~ 0.1 cm^-3), ~1.2e4 K, weakly ionized (n_e/n_H ~ 0.02-0.04), mostly
dusty (95-98%) gas carrying 4e-5 of the electrons (section 11, 2026-10-14).
`NeutralHydrogenAbundance == GrackleHI` for 100% of particles;
its maximum is 0.758-0.760 in the mass-fraction class (an HI *mass* fraction
capped at X_H) and 1.000 in the per-H class.

### 3.3 Dust tracks the class (not exactly)

Dust is a close but imperfect proxy for the class; the hydrogen sum
GrackleHI + GrackleHII is the exact indicator, and no correction should select
on dust. In the full m50n512 s50 snapshot, 435,386 dust-free particles (0.50%
of dust-free gas) use the per-H normalisation and 2 of 4.08e7 dusty particles use
mass fractions; in CAMELS-SIMBA CV_0 the counts are 20,430 (0.17%) and 5
(section 11). The SIMBA team made the same point. (Before 2026-10-14 this section
was titled "Dust determines the class".)

| Snapshot | P(per-H \| dust > 0) | P(per-H \| dust = 0) |
|---|---|---|
| m100n1024 z=0.49 | 1.00000 | 0.0053 |
| m100n1024 z=0 | 1.00000 | 0.0033 |
| m50n512 s50 | 1.00000 | 0.0050 |
| m50n512 s50nox / nojet / noagn / nofb | 1.0000 each | 0.0054 / 0.0064 / 0.0072 / 0.0002 |

The class also tracks metallicity (P(per-H | Z>0) = 0.97), because almost all
enriched gas carries dust. The mechanism is not visible in public code; one
plausible reading is that SIMBA's dust routine (Li, Narayanan & Davé 2019)
computes the gas state with GIZMO's native solver, which writes n_e/n_H and
per-H abundances. This is an inference.

### 3.4 Species-independent test: hot gas

Gas hot enough to be collisionally fully ionized must have n_e/n_H = 1.158.
T_min below uses the lowest possible mu (0.588), so it is a lower bound.

| Snapshot | Gas with T_min > 3.2e6 K | EA = 0.880 | of which dusty | EA = 1.158 | of which dusty |
|---|---|---|---|---|---|
| m100n1024 z=0.49 | 23.6% of gas mass | 49.7% | 0.00% | 49.1% | 99.3% |
| m100n1024 z=0 | 24.8% | 39.0% | 0.00% | 60.2% | 99.7% |
| m50n512 s50 | 21.2% | 45.0% | 0.00% | 53.6% | 99.5% |

Half of SIMBA's 10^6.5-10^7 K gas reports EA = 0.880, which cannot be n_e/n_H
for gas that hot, and every one of those particles is dust-free. 0.880 is
exactly X_H + Y/2, the fully ionized value in Grackle's normalisation.

### 3.5 Global budgets

| Run (z=0.49 unless noted) | ionized/gas as read | ionized/gas corrected | Sum(n_e) as read / corrected | tSZ Sum(n_e T_e) as read / corrected |
|---|---|---|---|---|
| m100n1024 s50 | 0.795 | 0.955 | 0.834 | 0.944 |
| m100n1024 s50, z=0 | 0.828 | 0.974 | n/c | n/c |
| m50n512 s50 | 0.799 | 0.955 | 0.837 | 0.943 |
| m50n512 s50nox | 0.798 | 0.947 | 0.843 | 0.950 |
| m50n512 s50nojet | 0.798 | 0.934 | 0.854 | 0.976 |
| m50n512 s50noagn | 0.774 | 0.931 | 0.831 | 0.966 |
| m50n512 s50nofb | 0.775 | 0.933 | 0.830 | 0.983 |

m50n512 variants and the tau/tSZ columns use a 5e6-particle block subsample; the
m100n1024 and m50n512 s50 budgets agree between block, random and full-snapshot
passes to better than 0.002. "ionized/gas" is the pipeline definition
M_ion = N_e m_p mu_e with mu_e = 2/(1+X_H).

For reference, the pipeline gives 0.983 for TNG300-1 (0.994 on 12 random chunk
files of snapshot 67) and 0.982 for CAMELS-TNG CV_0.

### 3.6 Where the correction matters (m100n1024 z=0.49, by density)

| n_H [cm^-3] | gas mass share | per-H share | ionized as read | ionized corrected | tau as read/corr | tSZ as read/corr | in FoF groups |
|---|---|---|---|---|---|---|---|
| < 1e-6 | 0.287 | 0.162 | 0.776 | 0.971 | 0.800 | 0.912 | 0.003 |
| 1e-6 - 1e-5 | 0.427 | 0.248 | 0.807 | 0.983 | 0.820 | 0.921 | 0.098 |
| 1e-5 - 1e-4 | 0.182 | 0.409 | 0.849 | 0.988 | 0.859 | 0.950 | 0.367 |
| 1e-4 - 1e-3 | 0.056 | 0.737 | 0.927 | 0.989 | 0.938 | 0.986 | 0.963 |
| 1e-3 - 1e-2 | 0.015 | 0.785 | 0.851 | 0.893 | 0.953 | 0.999 | 1.000 |
| 1e-2 - 0.1 | 0.011 | 0.782 | 0.137 | 0.144 | 0.946 | 0.993 | 1.000 |
| > 0.1 | 0.021 | 0.978 | 0.339 | 0.339 | 0.999 | 0.999 | 1.000 |

The mis-read particles are pristine (dust-free, Z=0) gas that has never been
through a galaxy, so they dominate the diffuse IGM. This is why SIMBA's
"neutral gas" appeared to sit far from clusters: it follows enrichment history,
not ionization state.

### 3.7 What remains non-ionized after the correction (4.4% of gas, m100n1024)

- ~50%: HI-dominated dense gas (genuinely neutral);
- ~5%: star-forming gas;
- ~40%: ~10^4 K low-density dust-free gas with H fully ionized but He only
  singly ionized (EA = 0.820). This is consistent with Grackle
  `self_shielding_method=3`, which sets HeII photoionization to zero; gas at the
  same temperature in the per-H class shows He fully ionized. Treat this piece as
  a model choice, not robust physics (inference).

H2 (`FractionH2`) is 0.6% of gas mass; HI (convention-aware) 1.8%.

### 3.8 Outliers

Scanning 1e8 particles of m100n1024 z=0.49 found 16 particles (1.6e-7 by
number, 2.3e-7 by mass) where EA/(HI+HII) exceeds 1.158: jet-kicked, currently
decoupled particles (`NWindLaunches` >= 1000, `DelayTime` > 0) with EA = 1.07895
(a per-H value) but mass-fraction species, and a few ~1e9 K, extremely low
density particles with trace dust. The cap at 1 + 2y handles them.

The full m50n512 s50 census (section 11.3) refines this: 11 particles exceed the
cap. 2 of them (the only ones above 1.5) carry a per-H EA of 1.158 with
mass-fraction species that also say fully ionized, so both corrections give
1.158. The other 9 are jet-launched particles with EA = 1.0789. The cap does not
catch every jet-launched particle: 16 more have per-H species, where neither
correction is right (section 11.3).

## 4. Controls

The same hot-gas test on TNG data, where EA is documented as n_e/n_H:

| Snapshot | T_min > 3.2e6 K: EA p1 / p50 / p99 | fraction at EA = 0.880 |
|---|---|---|
| CAMELS-TNG CV_0, z=0.47 (full) | 1.1579 / 1.160 / 1.169 | 0.0000 |
| TNG300-1 snap 67 (12 random chunk files, 2.9e8 particles) | 1.1579 / 1.1603 / 1.1674 | 0.0000 |

Whole-snapshot EA percentiles (1, 5, 25, 50, 75, 95, 99) for CAMELS-TNG:
0.359, 1.151, 1.1575, 1.1578, 1.1585, 1.1619, 1.1653. There is no 0.88 or 0.82
plateau anywhere in TNG.

## 5. CAMELS-SIMBA

Downloaded (approved 2026-09-22) from
`https://users.flatironinstitute.org/~camels/Sims/`, size-checked and opened with
h5py; kept at `/pscratch/sd/r/rhliu/simulations/CAMELS/`:

- `SIMBA/L25n256/CV/CV_0/snapshot_074.hdf5` (z=0.47, 1.75 GB)
- `SIMBA/L25n256/CV/CV_0/snapshot_090.hdf5` (z=0, 1.80 GB)
- `SIMBA/L25n256/1P/1P_p6_2/snapshot_074.hdf5` (z=0.47, `BH_QUENCH_JET` 1.4e4 vs 7000 km/s fiducial, 1.72 GB)
- `IllustrisTNG/L25n256/CV/CV_0/snapshot_074.hdf5` (z=0.47, control, 2.60 GB)

CAMELS-SIMBA carries the same gas fields (`Dust_Masses`, `Grackle*`,
`ElectronAbundance`, `NeutralHydrogenAbundance`). Full-snapshot results:

| Snapshot | Mass-frac class | P(per-H \| dust) | Hot (T_min>3.2e6 K) at EA=0.880 (dusty) | ionized/gas as read -> corrected | electrons as read/corr | T as read/corr (ionized, mass-frac class) | tSZ-like as read/corr |
|---|---|---|---|---|---|---|---|
| CV_0 z=0.47 | 76.3% | 1.00000 | 61.3% (0.00%) | 0.782 -> 0.960 | 0.815 | 1.142 | 0.915 |
| CV_0 z=0 | 72.5% | 1.00000 | 53.8% (0.00%) | 0.803 -> 0.973 | 0.825 | 1.142 | 0.920 |
| 1P_p6_2 z=0.47 | 78.5% | 0.99999 | 74.2% (0.00%) | 0.779 -> 0.963 | 0.809 | 1.142 | 0.906 |

The CAMELS library (`library/camels_library/camels_library.py`, functions
`electron_density` and `temperature`) applies `n_e = 0.76*ne*rho/m_p` and
`T = u(1+4y)/(1+y+ne)` to every suite, so CAMELS-SIMBA electron densities from
the library are ~18-19% low and temperatures of ionized dust-free gas 14% high.

## 6. How others read the field

Items marked (v) were read directly during this investigation; the rest come
from two literature searches (2026-09-22/23) and were not individually re-read.

**SIMBA-team papers and tools: every stated convention is n_e/n_H.**

- Yang, Cai, Cui, Davé, Peacock, Sorini 2022 (arXiv:2202.11430) (v): "the
  electron abundance per gas particle defined as the fractional electron number
  relative to the total hydrogen number".
- Dong et al. 2025 (arXiv:2507.16115, with Davé): N_e = eta_e X_H M_gas/m_p.
- Nicola et al. 2022 (arXiv:2201.04142, with Davé and Anglés-Alcázar): electrons
  from "volume, density and ElectronAbundance (see [TNG specifications])",
  applied to CAMELS-SIMBA too.
- Cui et al. 2018 (PyMSZ): "N_e is the number of ionized electrons per hydrogen
  particle"; PyMSZ `load_data.py`: "electron number = M*X/m_H*NE".
- pygad (v): `snapshot/derive_rules.py` lines 84-86 treat `ne` as electrons per
  H; `analysis/properties.py` line 504 carries R. Davé's 2018 comment and the
  X-ray emission code used by Robson & Davé 2020/21 multiplies by `ne` per H.
- caesar (v): `hydrogen_mass_calc/hydrogen_mass_calc.pyx` lines 182, 215:
  `fHI = gnh[i]`, `HImass[i] = fHI * XH * gmass[i]`, i.e.
  `NeutralHydrogenAbundance` read as per H. For dust-free gas it is a mass
  fraction (section 3.2), so caesar's HI masses for that gas would be low by
  X_H (not checked further).
- pygadgetreader, XIGrM, The300 GIZMO-SIMBA scripts: mu = 4/(1+3X+4X ne).
- No erratum or statement of a different convention was found.

**Independent works: all read n_e/n_H or avoid the field.**

- Lee, Coulton, Thiele & Ho 2022 (arXiv:2205.01710) (v): "We prepare the data
  as in the IllustrisTNG simulation" for SIMBA.
- Thiele et al. / Wadekar et al. (group_particles `examples/CAMELS_profiles.cpp`
  line 150, compiled with `FOR_SIMBA`) (v): `rho_e = x * XH * m`.
- CAMELS library (v), yt GIZMO frontend (`ElectronAbundance * H_nuclei_density`) (v).
- Guo & Lee 2025 (CAMELS library), Medlock et al. 2024/25 (yt), Moser et al.
  2022 (n_e/n_H for T; full ionization for SZ), Pandey et al. 2023 and Bigwood
  et al. 2025 (full ionization; unaffected for n_e).
- Sokoliuk et al. 2025 (arXiv:2510.07259) (v) flags EA as "unreliable for gas
  cells with [SFR>0]": a different issue (multiphase star-forming gas).
- No GitHub issue, caveat or erratum about this was found (yt, caesar, CAMELS,
  pygad searched).

## 7. Interpretation and confidence

Established by the data (high confidence):
- For dust-free SIMBA and CAMELS-SIMBA gas, `ElectronAbundance` equals
  n_e m_p/rho; for dusty gas it equals n_e/n_H. The species fields follow the
  same split. This holds in 9 SIMBA snapshots from two data releases, at z=0.47-0.49
  and z=0, and across feedback variants, and never appears in TNG. Dust tracks
  the split closely but not exactly (section 3.3).
- `EA/(GrackleHI+GrackleHII)`, capped at 1.158, recovers n_e/n_H for all but
  a few 1e-7 of the gas mass, and so does the species-only formula the SIMBA
  team recommends. The exceptions are jet-launched particles (sections 3.8 and
  11.3).
- Tentatively confirmed by the SIMBA team (2026-10-14, section 11).

Supported but not proven:
- The mechanism: SIMBA's GIZMO fork writes Grackle's `e_density/rho` into `Ne`
  for Grackle-evolved particles (as in the pre-2019 public fork), while another
  code path, correlated with dust, writes GIZMO-native n_e/n_H and per-H
  abundances. SIMBA's GIZMO source is private.
- Whether the simulation's own evolution is affected, as opposed to only the
  output. Grackle uses its normalisation consistently internally, so cooling of
  dust-free gas is probably fine; the dusty path is unknown.

Not a pipeline bug in the usual sense: `mapMaker` implements the documented
convention. The same reading is used by the SIMBA team's own published analyses
and by all independent works found.

Before any public claim: confirm with the SIMBA developers and the CAMELS team
(draft message: `simba_electron_abundance_message_draft.md`). The message was
sent; the SIMBA team's reply is summarised in section 11.

## 8. Implications for this repository and the papers

- `src/mapMaker.py`: SIMBA `ionized_gas`, `neutral_gas`, `tau`, `kSZ` and `tSZ`
  fields use EA as n_e/n_H. No code or cached field has been changed.
- Unbound gas paper: the claims attributing SIMBA's ionized-vs-baryon offset to
  on-the-fly H2/neutral partitioning (abstract; main.tex ~114, ~546, ~712 result
  (ii), ~798) are not supported: after correction SIMBA's non-ionized gas is 4.4%
  (vs 20.5%), of which H2 is 0.6 points. SIMBA tau/kSZ amplitudes rise by ~5-6%
  in halo gas and ~18-20% in the diffuse IGM; tSZ changes little in haloes
  (<1.5%) and ~8-9% in the IGM.
- Profile-level impact on the paper figures and on the lensing paper
  (Figs 14 and 16): see section 10.

## 9. Reproduction

Minimal test (runs on a login node in seconds for a CAMELS snapshot):

```python
import h5py, hdf5plugin, numpy as np
fn = '/pscratch/sd/r/rhliu/simulations/CAMELS/SIMBA/L25n256/CV/CV_0/snapshot_074.hdf5'
with h5py.File(fn, 'r') as f:
    g = f['PartType0']
    EA, u = g['ElectronAbundance'][:], g['InternalEnergy'][:].astype(float)
    Hs = g['GrackleHI'][:] + g['GrackleHII'][:]
    dust = g['Dust_Masses'][:] > 0
T_min = (2/3) * u * 1e10 * 1.6726e-24 * 0.588 / 1.3807e-16   # K, lower bound (mu = 0.588)
hot = T_min > 10**6.5
print('hot gas at EA=0.880:', np.mean(np.abs(EA[hot] - 0.880) < 0.005),
      ' dusty among them:', dust[hot][np.abs(EA[hot] - 0.880) < 0.005].mean())
print('per-H class (HI+HII=1) given dust:', np.mean(np.abs(Hs[dust] - 1) < 1e-3))
ne_nH = np.minimum(EA / Hs, 1 + 2 * 0.24 / 3.04)             # corrected n_e/n_H
```

## 10. Impact on the paper figures (in-memory study, 2026-09-23)

Scripts (new, `scripts/simba_test/`; nothing in `src/` or the paper scripts was
edited): `simba_ea_correction.py` (wraps `mapMaker.load_subset` so the existing
field builders see the corrected EA, and disables every `save_data`),
`lensing_fig14_16_simbaEA.py`, `unbound_simbaEA_difference.py`,
`runINT_simbaEA.sh`. Run on interactive CPU nodes, account desi: smoke test job
58783516 (7 min, 43 GB peak) and full run job 58783636 (30.5 min, 182 GB peak).
Logs: `Outputs_Perlmutter/simbaEA-58783636_*.out`. Outputs (figures + `.npz` of
the plotted curves): `figures/2026-09/09-23/simba_test/`. Scratch caches were
read, never written.

Validation (all exact):
- rebuilding a SIMBA map in memory with the wrapper disabled reproduces the
  cached map bit for bit (m50n512 s50noagn and m100n1024 s50: max |diff| = 0);
- SIMBA as read, restacked by the new script, reproduces the published zenodo
  curves of lensing Figs 14a/b and 16a/b, SIMBA's cached beam factors, and the
  published beam-compensated data points (max relative difference 0).

### 10.1 Lensing paper, Figs 14 and 16 (SIMBA m100n1024 s50)

Other simulations and the Fig 14 data are taken unchanged from
`~/projects/DESIxHSC-Lensing/zenodo/`. Ratios are corrected / as read, from
theta = 1' to 6':

| Panel | SHAM haloes | SIMBA f_gas (Fig 14) | SIMBA f_gas (Fig 16) | SIMBA beam factor | Fig 16 data points | Fig 16 detection SNR |
|---|---|---|---|---|---|---|
| a (z=0.26, snap 136) | 1000 | 1.005 -> 1.044 | 1.000 -> 1.044 | 1.005 -> 1.000 | 0.9992 - 1.0002 | 17.19 -> 17.20 |
| b (z=0.5, snap 125) | 500 | 1.025 -> 1.078 | 1.019 -> 1.078 | 1.006 -> 0.998 | 0.9990 - 1.0003 | 13.81 -> 13.83 |

SIMBA's curve rises by up to ~4% (z=0.26) and ~8% (z=0.5) at 6'; its ranking
among the six simulations does not change. The beam compensation is
insensitive to the correction (it is a ratio of the same component).

### 10.2 Unbound gas paper (z = 0.5)

Corrected / as read, innermost -> outermost radius (3D: 200 -> 4000 ckpc/h;
2D: the same comoving range in arcmin; tau and tSZ CAP: 1' -> 6'):

| Run | ion/gas (global) | 3D f_ion (Fig 2) | 2D cumulative f_ion (Fig 2) | 2D CAP f_ion (Fig 2) | tau CAP (Fig 7, unmasked) | tSZ CAP (Fig 11, unmasked) |
|---|---|---|---|---|---|---|
| m100n1024 s50 | 0.795 -> 0.955 | 1.026 -> 1.135 | 1.134 -> 1.181 | 1.033 -> 1.113 | 1.016 -> 1.072 | 1.014 -> 1.042 |
| m50n512 s50noagn | 0.775 -> 0.932 | 1.016 -> 1.143 | 1.087 -> 1.173 | 1.030 -> 1.106 | 1.032 -> 1.077 | n/a |
| m50n512 s50nox | 0.803 -> 0.948 | 1.006 -> 1.118 | 1.082 -> 1.149 | 1.006 -> 1.086 | 1.019 -> 1.055 | n/a |
| m50n512 s50nofb | 0.772 -> 0.932 | 1.001 -> 1.129 | 1.064 -> 1.171 | 1.002 -> 1.087 | 1.007 -> 1.047 | n/a |

Non-ionized gas as a fraction of all baryons in m100n1024 drops from 19.8% to
4.3%. Near halo centres the change is small (a few per cent, because halo gas is
mostly dust-bearing and already n_e/n_H); it grows outward to 11-18% at
4 Mpc/h, where pristine gas dominates. The masked (unbound-only) columns of
Figs 7 and 11 were not recomputed; since the mis-read gas is mostly diffuse and
far from haloes, those columns are expected to change more than the unmasked
ones (not verified).

## 11. SIMBA team reply and the species-ratio correction (2026-10-14)

### 11.1 The reply (paraphrased)

The message in `simba_electron_abundance_message_draft.md` was sent. The SIMBA
team replied (received by 2026-10-14) and tentatively confirmed the bug:

- `ElectronAbundance` is biased. The Grackle species and EA are stored in two
  conventions, either mass fractions of the gas (HI + HII = 0.76) or numbers per
  H atom (HI + HII = 1). The convention switches from particle to particle, but
  within one particle the Grackle species share it.
- Their workaround uses only ratios of the Grackle fields:
  n_e/n_H = HII/(HI+HII) + f_HeH (HeII + 2 HeIII)/(HeI + HeII + HeIII), with
  f_HeH = (1 - X)/(4X). It cannot exceed 1 + 2 f_HeH = 1.157895, needs no cap,
  and does not use EA.
- On our correction: EA/(HI+HII) does effectively the same thing whenever EA and
  the species share a convention. They find some particles with a per-H EA but
  mass-fraction H species, where the ratio exceeds 1.5. They guessed that is why
  we cap.
- Dust is connected to the convention but not exactly. Some dust-free
  particles use per-H, and some dusty particles use mass fractions. The
  hydrogen sum, not dust, is the exact indicator.
- Any electron density computed from the raw EA is biased. So is any
  calculation that uses the Grackle fields without accounting for the
  normalisation.

### 11.2 Checks of the reply

All checks were read-only and run in memory from stdin, with no files saved.

| Snapshot | Particles used |
|---|---|
| CAMELS-SIMBA CV_0 snapshot 074 | full, 1.6e7 |
| SIMBA m50n512 s50 snapshot 125 | full, 1.28e8, streamed on a login node; plus a 5e6 subsample (20 random blocks of 250k, seed 1) |
| SIMBA m100n1024 s50 snapshot 125 | 5e6 subsample (20 random blocks of 250k, seed 1) |

A full m100n1024 pass was not possible: the interactive allocation (job
59794313) was revoked because every Perlmutter partition was down.

**Algebra.** In the mass-fraction class, H = 0.76(1-Z) and He = 0.24(1-Z)
(section 3.1), and EA = HII + HeII/4 + HeIII/2. So
EA/(HI+HII) = HII/H + y (HeII + 2 HeIII)/He, which is the species formula
exactly, metals included. In the per-H class, HI + HII = 1, so the two agree
exactly when EA = HII + HeII + 2 HeIII.

**Budgets and the hot-gas test.** "Clamp" is the capped EA/(HI+HII); "species"
is the SIMBA team's formula. Both corrections give the same ionized/gas.

| Snapshot | ionized/gas, as read -> corrected | N_e, clamp / species | Σ n_e T_e, clamp / species | T_min > 10^6.5 K gas off 1.158 by > 0.5%: as read / clamp / species |
|---|---|---|---|---|
| CAMELS-SIMBA CV_0 (full) | 0.7819 -> 0.9595 | 1.000000 | 1.000000 | 63.3% / 0.045% / 0.045% |
| m50n512 s50 (full) | 0.7994 -> 0.9557 | 1.0000006 | 0.9999998 | 46.4% / 0.033% / 0.033% (8,925 / 8,927 of 2.70e7 particles) |
| m100n1024 s50 (5e6 subsample) | 0.7956 -> 0.9566 | 1.000001 | 1.000000 | 48.3% / 0.033% / 0.033% |

Σ n_e T_e uses each correction's own n_e/n_H in mu, as `make_sz_field` does.

**Density bins and haloes.** We split the gas into log n_H bins (< -6 up to
> -1) and by inside versus outside FoF groups.

- N_e (clamp / species) is within 1e-4 of 1 in every density bin of all three
  runs.
- Σ n_e T_e is within 1e-5 of 1 in every bin except one. In the densest bin of
  the m100n1024 subsample (n_H > 0.1 cm^-3) it is 1.0008, set by a single
  jet-launched particle (section 11.3).
- Inside and outside FoF groups, the two corrections agree to 4e-6.
- Summed over the whole snapshot, Σ|clamp - species| M / Σ species M is 2.5e-6
  (CAMELS), 2.7e-6 (m50n512, full) and 2.8e-6 (m100n1024 subsample).

### 11.3 Where the two corrections differ (full m50n512 s50 census)

**How each particle's EA convention was classified.** For particles with
species n_e/n_H x_sp > 0.02, compute r = EA / x_sp:

- per-H if |r - 1| < 1%;
- mass fraction if r agrees with HI + HII to 1% (mass-fraction species), or if
  0.62 < r < 0.77 (per-H species);
- "neither" otherwise.

| Species class | EA convention | Particles | Gas mass share | Electron share |
|---|---|---|---|---|
| per-H | per-H | 40,007,559 | 0.318 | 0.319 |
| per-H | neither | 15 | 1.5e-7 | 1.4e-7 |
| per-H | x_sp <= 0.02 | 1,184,027 | 9.6e-3 | 5.2e-5 |
| mass fraction | per-H | 2 | 2.6e-8 | 2.7e-8 |
| mass fraction | mass fraction | 87,022,649 | 0.672 | 0.681 |
| mass fraction | neither | 7 | 1.0e-7 | 2.5e-8 |
| mass fraction | x_sp <= 0.02 | 6,745 | 5.9e-5 | 6.4e-7 |

No particle pairs a mass-fraction EA with per-H species, here or in the full
CAMELS-SIMBA CV_0 snapshot. That is the one mismatch no cap can detect: it would
leave the capped ratio 24% low.

**The cap and the >1.5 ratios.** EA/(HI+HII) exceeds the cap for 11 particles.

- **2 particles above 1.5** (maximum 1.555): they carry a per-H EA of 1.158 with
  mass-fraction species that also say fully ionized. Both corrections give
  1.158 there. CAMELS-SIMBA CV_0 has one such particle (ratio 1.553; it is
  dusty, one of the 5 dusty mass-fraction particles).
- **The other 9** are jet-launched particles, described below.

**The 25 particles that disagree.** 25 particles (2.9e-7 of the gas mass) differ
by more than 0.01 in n_e/n_H between the two corrections. All 25 have:

- EA = 1.0789 exactly, which is 1 + y: H ionized and He singly ionized,
  in the per-H convention;
- `NWindLaunches` >= 1000 (jet-launched) and `DelayTime` > 0 (hydrodynamically
  decoupled);
- no dust, no star formation, and FoF group membership;
- T_min = 10^5.2-10^6.7 K.

What each correction returns:

| Species class | Particles | Species formula | Capped ratio |
|---|---|---|---|
| mass fraction | 9 | 0.0003-0.37 | 1.158 (the cap) |
| per-H | 16 | 0.015-1.15 | 1.079 (EA itself) |

Gas this hot should be fully ionized, at ~1.158. The species look like they were
carried over from before the launch, and EA = 1 + y looks like a fixed value set
at launch (both inferences; SIMBA's GIZMO source is private).

Neither correction is reliable for these particles:

- the species formula gives 0.0003-1.15, mostly well below 1.158;
- the capped ratio gives the expected 1.158 only for the 9 with mass-fraction
  species, and only because it hits the cap; for the other 16 it returns the
  fixed EA = 1.079.

At 2.9e-7 of the gas mass they affect none of our results.

### 11.4 Assessment

- The two corrections are the same quantity wherever EA and the species share a
  normalisation.
- The global budgets above agree to 1e-6. Per density bin, they agree to 1e-4,
  except one bin at 8e-4 set by a single jet-launched particle.
- The profile-level impacts in section 10 should therefore carry over unchanged
  to the species formula. This is an inference from the budget agreement; the
  figures were not re-run.
- Use the species formula for any pipeline fix:
  - it needs no cap and does not depend on EA;
  - it is what the SIMBA team recommends;
  - HI/(HI+HII) from the same fields gives a neutral H fraction that does not
    depend on the normalisation, for `neutral_gas`.
- Cost: the species formula reads HI, HII, HeI, HeII and HeIII instead of EA.
  That is 2 more float32 fields per particle than the capped ratio (~8 GB per
  full m100n1024 pass) and 4 more than the current EA-only read.
- `src/mapMaker.py`, `src/loadIO.py` and the scratch caches are unchanged. Any
  fix needs explicit approval.

### 11.5 Not checked

- the full m100n1024 snapshot (blocked by the partition outage);
- the species formula at z = 0, in the four other m50n512 variants, and on
  CAMELS-SIMBA snapshot 090 or 1P_p6_2;
- the masked Fig 7/11 columns, and any profile-level figure under the species
  formula.

### 11.6 Reproduction

```python
import h5py, hdf5plugin, numpy as np
fn = '/pscratch/sd/r/rhliu/simulations/CAMELS/SIMBA/L25n256/CV/CV_0/snapshot_074.hdf5'
with h5py.File(fn, 'r') as f:
    g = f['PartType0']
    EA, M = g['ElectronAbundance'][:].astype(float), g['Masses'][:].astype(float)
    HI, HII = g['GrackleHI'][:].astype(float), g['GrackleHII'][:].astype(float)
    HeI, HeII, HeIII = (g[k][:].astype(float) for k in ('GrackleHeI', 'GrackleHeII', 'GrackleHeIII'))
y = 0.24 / (4 * 0.76)
x_species = HII / (HI + HII) + y * (HeII + 2 * HeIII) / (HeI + HeII + HeIII)   # SIMBA team
x_capped = np.minimum(EA / (HI + HII), 1 + 2 * y)                             # section 3
print(np.sum(x_capped * M) / np.sum(x_species * M))                       # 1.000000
print(np.sum(np.abs(x_capped - x_species) * M) / np.sum(x_species * M))   # 2.5e-06
print(x_species.max())                                                    # 1.157895
```
