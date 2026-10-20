# Bug report: SIMBA's `ElectronAbundance` field mixes two normalisations

**23 September 2026, updated 14 October 2026.** Status: the SIMBA team has
tentatively confirmed the bug and recommends a correction built from the Grackle
species alone (section 3). It agrees with our original correction to $10^{-6}$
in every global budget we checked. The CAMELS team has not replied. Full record:
`simba_electron_abundance_report.md` (section 11 covers the reply).

## Summary

- In the SIMBA and CAMELS-SIMBA snapshots, `PartType0/ElectronAbundance` (EA
  below) is documented as **n_e/n_H**, the number of free electrons per hydrogen
  nucleus. Every tool and paper we found reads it that way.
- For about two thirds of SIMBA gas particles, EA instead stores Grackle's
  internal normalisation, **n_e m_p/ρ = X_H · n_e/n_H**, which is 24% lower. The
  share is 63–69% in SIMBA m100n1024 and m50n512 and 72–79% in CAMELS-SIMBA.
- **Each particle's Grackle species fields use the same normalisation as its
  EA.** The hydrogen sum GrackleHI + GrackleHII (X_H(1 − Z) or 1) identifies the
  class exactly. Dust tracks it closely but not exactly: almost all dusty
  particles store n_e/n_H, and more than 99% of dust-free particles store the
  Grackle value (section 2.3).
- Reading EA as n_e/n_H everywhere undercounts SIMBA's free electrons by about
  17%, mostly in the diffuse IGM. It also biases temperatures derived from EA
  high by 14% in the affected ionized gas.
- The SIMBA team's species-ratio formula recovers n_e/n_H for all but about
  $3 \times 10^{-7}$ of the gas mass, without using EA:

```
n_e/n_H = GrackleHII / (GrackleHI + GrackleHII)
        + y (GrackleHeII + 2 GrackleHeIII) / (GrackleHeI + GrackleHeII + GrackleHeIII)

y = n_He/n_H = (1 - X_H) / (4 X_H) = 0.0789   (X_H = 0.76; maximum 1 + 2y = 1.158)
```

  Our original correction, `min(EA / (GrackleHI + GrackleHII), 1 + 2y)`, gives
  the same value wherever EA and the species share a normalisation (section 3).

We found this while checking the ionized-gas budgets for a kSZ/tSZ stacking
analysis. Our pipeline also reads EA as n_e/n_H, and with that reading about
20% of SIMBA's gas appeared non-ionized, compared with under 2% in IllustrisTNG.

## 1. The problem

The GIZMO user guide defines EA as "the mean free-electron number per proton
(hydrogen nucleon); averaged over the mass of the gas particle", and IllustrisTNG
uses the same definition. SIMBA uses Grackle for cooling and chemistry, and
Grackle normalises electrons differently: its electron "density" is the electron
number density times the proton mass. Grackle's `e_density/density` is therefore
n_e m_p/ρ = X_H · n_e/n_H. For the same ionization state, the two readings differ
by a factor of X_H = 0.76:

| Ionization state | n_e/n_H (documented) | n_e m_p/ρ (Grackle) |
|---|---|---|
| Fully ionized H and He | 1 + 2y = **1.158** | X_H + Y/2 = **0.880** |
| H ionized, He singly ionized | 1 + y = 1.079 | X_H + Y/4 = 0.820 |

Here X_H = 0.76 and Y = 0.24 are the primordial H and He mass fractions.

In the snapshots, SIMBA's gas particles fall into two classes. Dust tracks the
classes closely but not exactly; the exact indicator is the hydrogen sum
(section 2.3):

| | Mass-fraction class (≈ dust-free) | Per-H class (≈ dust-bearing) |
|---|---|---|
| `ElectronAbundance` | n_e m_p/ρ (Grackle) | n_e/n_H (documented) |
| `GrackleHI`, `GrackleHII`, `GrackleHe*` | mass fractions | abundances per H nucleus |
| `NeutralHydrogenAbundance` (= `GrackleHI`) | HI mass fraction | HI per H nucleus |
| Share of gas particles: SIMBA / CAMELS-SIMBA | 63–69% / 72–79% | 31–37% / 21–28% |

Nearly all metal-enriched gas carries dust. The misread particles are therefore
pristine gas that has never been through a galaxy, which makes up most of the
diffuse IGM. For the same reason, `NeutralHydrogenAbundance` is a mass fraction
in dust-free gas. HI masses computed as `NeutralHydrogenAbundance × X_H × mass`
therefore come out low by a factor of X_H for that gas. We have not checked this
side effect further.

## 2. Evidence

Snapshots checked (all from public releases):

- SIMBA m100n1024 s50 at z = 0.49 and z = 0;
- SIMBA m50n512 at z = 0.49, all five feedback variants (s50, s50nox, s50nojet,
  s50noagn, s50nofb);
- CAMELS-SIMBA L25n256 CV_0 at z = 0.47 and z = 0, and 1P_p6_2 at z = 0.47;
- controls: IllustrisTNG TNG300-1 at z = 0.5 and CAMELS-TNG CV_0 at z = 0.47.

The pattern appears in every SIMBA and CAMELS-SIMBA snapshot and in neither TNG
control.

### 2.1 Hot gas: the most direct test

Gas hot enough to be collisionally ionized must have n_e/n_H = 1.158, whatever
the species fields say. We select gas with T > $3.2 \times 10^{6}$ K. T is
computed with the lowest possible mean molecular weight (μ = 0.588), so the cut
is a lower bound on the true temperature.

| Snapshot | Hot gas at EA = 0.880 | …of which carries dust |
|---|---|---|
| SIMBA m100n1024, z = 0.49 | 49.7% | 0.00% |
| SIMBA m100n1024, z = 0 | 39.0% | 0.00% |
| SIMBA m50n512 s50, z = 0.49 | 45.0% | 0.00% |
| CAMELS-SIMBA CV_0, z = 0.47 | 61.3% | 0.00% |
| CAMELS-SIMBA CV_0, z = 0 | 53.8% | 0.00% |
| CAMELS-SIMBA 1P_p6_2, z = 0.47 | 74.2% | 0.00% |
| *TNG300-1, z = 0.5 (control)* | *0%* | |
| *CAMELS-TNG CV_0, z = 0.47 (control)* | *0%* | |

In SIMBA, 39–50% of the hot gas sits at exactly 0.880, and 54–74% in
CAMELS-SIMBA. That is the fully ionized value in Grackle's normalisation; it
cannot be n_e/n_H for gas this hot. None of these particles carries dust. The
remaining hot SIMBA gas sits at 1.158, and 99.3–99.7% of it carries dust. In
both TNG controls, the hot gas has EA between 1.158 and 1.17 (1st to 99th
percentile). TNG shows no 0.88 or 0.82 plateau anywhere.

### 2.2 EA is consistent with each particle's own species fields

The Grackle species fields (`GrackleHI`, `GrackleHII`, `GrackleHM`,
`GrackleHeI/II/III`) show the same two normalisations. No particle falls between
them:

- **Mass-fraction class:** HI + HII + HM = 0.76(1 − Z) and total He =
  0.24(1 − Z), so n_He/n_H = y exactly. For 100.000% of these particles,
  EA = HII + HeII/4 + HeIII/2, which is n_e m_p/ρ.
- **Per-H class:** HI + HII + HM = 1 exactly and total He = 0.0789 = y. For
  97.2–98.3% of these particles, EA = HII + HeII + 2 HeIII, which is n_e/n_H.
  Most of the rest are nearly neutral, where a relative tolerance is
  ill-defined. Excluding particles with n_e/n_H ≤ 0.02, 99.45% match at 0.2%
  and all but 15 of $4.0 \times 10^{7}$ match at 1% (m50n512 s50).

We checked this to 0.2% tolerance in SIMBA m100n1024 at z = 0.49 and z = 0 and in
m50n512 s50. The values are not random errors: each particle's EA agrees with its
own species fields. The snapshot simply mixes two conventions, one per class.
(Before 14 October the mass-fraction He total was given as 0.24.)

### 2.3 Dust tracks the class, but not exactly

- Dust-bearing particles fall in the per-H class with probability 1.0000 in all
  seven SIMBA snapshots, and 0.99999–1.00000 in the three CAMELS-SIMBA snapshots.
  The exceptions are few but real: 2 of $4.08 \times 10^{7}$ dusty particles in
  m50n512 s50, and 5 in CAMELS-SIMBA CV_0, use mass fractions.
- Dust-free particles fall in the per-H class with probability 0.0002–0.0072
  (435,386 particles in m50n512 s50).

So the dust flag should not be used to choose the convention. The hydrogen sum
is the exact indicator, as the SIMBA team also pointed out. Metallicity also
tracks the class (P(per-H | Z > 0) = 0.97), but less sharply than dust does.
(Before 14 October this section was titled "Dust decides the class".)

### 2.4 Electron budgets

If EA is read as n_e/n_H, SIMBA's ionized gas fraction falls well below TNG's.
After the correction, the two are comparable:

| Run | Ionized gas fraction, as read → corrected | Total free electrons, ratio as read / corrected | Σ n_e T_e (tSZ-like), ratio as read / corrected |
|---|---|---|---|
| SIMBA m100n1024 s50, z = 0.49 | 0.795 → 0.955 | 0.834 | 0.944 |
| SIMBA m50n512 s50, z = 0.49 | 0.799 → 0.955 | 0.837 | 0.943 |
| SIMBA m50n512, other four variants | 0.774–0.798 → 0.931–0.947 | 0.830–0.854 | 0.950–0.983 |
| CAMELS-SIMBA CV_0, z = 0.47 | 0.782 → 0.960 | 0.815 | 0.915 |
| *TNG300-1, z = 0.5 (reference)* | *0.983* | | |

The ionized gas fraction is the ionized gas mass implied by the electron count
(M_ion = N_e m_p μ_e, with μ_e = 2/(1 + X_H)), divided by the total gas mass.

The error depends on density because the misread particles are pristine gas. In
m100n1024 at z = 0.49:

- Below n_H = $10^{-5}\,\mathrm{cm^{-3}}$ (the diffuse IGM), the electron count
  is 18–20% low.
- Between n_H = $10^{-4}$ and $10^{-1}\,\mathrm{cm^{-3}}$, where 96–100% of the
  gas lies in FoF groups, the count is 5–6% low.
- The tSZ-like sum Σ n_e T_e is off by 8–9% in the IGM and by at most 1.5% in
  haloes. There, the temperature bias partly cancels the electron deficit.
- Temperatures derived from EA through μ = 4/(1 + 3X_H + 4X_H · EA) are 14% too
  high (a factor of 1.142) for ionized dust-free gas.
- The non-ionized share of SIMBA's gas falls from 20.5% to 4.4% after the
  correction.

### 2.5 How the field is read elsewhere

Every convention we found, in code or stated in papers, is n_e/n_H. That
includes the SIMBA team's own:

- **SIMBA-team papers and tools.** Yang et al. 2022 (arXiv:2202.11430) define EA
  as "the fractional electron number relative to the total hydrogen number".
  pygad and caesar treat EA and `NeutralHydrogenAbundance` as per-H quantities,
  and so do PyMSZ (Cui et al. 2018) and the analyses in Nicola et al. 2022 and
  Dong et al. 2025.
- **General tools.** The CAMELS library (`electron_density`, `temperature`) and
  the yt GIZMO frontend apply n_e/n_H to every simulation suite. CAMELS-SIMBA
  electron densities from the library are therefore about 18–19% low, and
  temperatures of ionized dust-free gas 14% high.
- **Independent SZ work on SIMBA.** Lee et al. 2022 (arXiv:2205.01710) and the
  `group_particles` profile code (Thiele, Wadekar et al.) use the TNG convention.
- We found no erratum, GitHub issue or caveat about this in yt, caesar, CAMELS or
  pygad.

We read Yang et al. 2022, Lee et al. 2022, and the code of the CAMELS library,
yt, pygad, caesar and `group_particles` directly. The other attributions come
from a literature search and were not individually re-checked.

## 3. Correction

**Recommended (the SIMBA team's species-ratio formula):**

```
n_e/n_H = GrackleHII / (GrackleHI + GrackleHII)
        + y (GrackleHeII + 2 GrackleHeIII) / (GrackleHeI + GrackleHeII + GrackleHeIII)
```

- It uses only ratios within one particle's species, so the normalisation
  cancels in both classes.
- n_He/n_H = y exactly in both classes, because the mass-fraction class has
  He = 0.24(1 − Z) alongside H = 0.76(1 − Z).
- It cannot exceed 1 + 2y = 1.157895, so it needs no cap, and it does not use EA.

**Our original correction (equivalent wherever EA and the species share a
normalisation):**

```
n_e/n_H = min( ElectronAbundance / (GrackleHI + GrackleHII),  1.158 )
```

- **Per-H class:** the divisor is 1, so EA is unchanged.
- **Mass-fraction class:** the divisor is the hydrogen mass fraction,
  X_H(1 − Z). Dividing by it converts n_e m_p/ρ into n_e/n_H, metallicity
  dependence included.
- **Cap at the fully ionized value:** this affects about $10^{-7}$ of the gas
  mass.
  - In m100n1024 it catches 16 of $10^{8}$ particles scanned. Most are
    jet-kicked particles that are currently hydrodynamically decoupled. They
    carry a per-H EA (1.079) but mass-fraction species. The rest are a few
    ~$10^{9}$ K, very low-density particles with trace dust.
  - CAMELS-SIMBA CV_0 gives the same scale ($1.2 \times 10^{-7}$ of the gas
    mass above the cap by more than rounding).

**How the two compare (14 October).** We checked CAMELS-SIMBA CV_0 and SIMBA
m50n512 s50 in full, and a $5 \times 10^{6}$-particle subsample of SIMBA
m100n1024 s50.

| Run | Ionized gas fraction, as read → corrected (both corrections) | Electrons, original / species formula |
|---|---|---|
| CAMELS-SIMBA CV_0, z = 0.47 | 0.782 → 0.960 | 1.000000 |
| SIMBA m50n512 s50, z = 0.49 | 0.799 → 0.956 | 1.0000006 |
| SIMBA m100n1024 s50, z = 0.49 (subsample) | 0.796 → 0.957 | 1.000001 |

The tSZ-like sum Σ n_e T_e agrees to $10^{-6}$ as well.

In the full m50n512 s50 snapshot the two disagree (by more than 0.01) for only 25
particles, $2.9 \times 10^{-7}$ of the gas mass. All 25 are jet-launched,
currently decoupled, dust-free particles at T > $10^{5.2}$ K with EA = 1.0789
(= 1 + y) exactly:

- where their species are mass fractions, the species formula gives 0.0003–0.37
  and the capped ratio gives 1.158;
- where their species are per H, the species formula gives 0.015–1.15 and the
  capped ratio gives 1.079.

Gas this hot should be at ~1.158. The species formula falls well short for
most of them, and the capped ratio reaches 1.158 only where it hits the cap.
Neither is reliable for these particles. The species look stale from before the
launch (an inference).

No particle combines a mass-fraction EA with per-H species. That is the one
mismatch the cap could not catch. The SIMBA team reported particles with
EA/(HI + HII) above 1.5. We find 2 in m50n512 s50 and 1 in CAMELS-SIMBA CV_0.
All have a per-H EA of 1.158 with mass-fraction species, and both corrections
give 1.158 there.

We recommend the species formula: no cap, no dependence on EA, and it is what
the SIMBA team recommends. HI/(HI + HII) from the same fields gives a
normalisation-free neutral fraction.

## 4. Confidence and open questions

**Established by the data (high confidence)**

- For dust-free gas, EA = n_e m_p/ρ. For dust-bearing gas, EA = n_e/n_H, with
  the small exceptions in section 2.3. The species fields follow the same split,
  and the hydrogen sum identifies it exactly.
- This holds at z ≈ 0.5 and z = 0, across feedback variants, and in both the
  original SIMBA runs and CAMELS-SIMBA. It never appears in TNG.
- Both corrections in section 3 recover n_e/n_H for all but a few $10^{-7}$ of
  the gas mass. The exceptions are jet-launched particles.
- The SIMBA team has tentatively confirmed the bug (reply received by
  14 October 2026).

**Plausible but unconfirmed**

- *Mechanism.* SIMBA's version of GIZMO is private.
  - A public fork from the same pre-2019 GIZMO lineage stores Grackle's
    `e_density/density` directly in the electron-abundance variable
    (`*ne_guess = my_fields.e_density[0] / my_fields.density[0]`).
  - GIZMO's native cooling solver writes n_e/n_H instead.
  - Our working hypothesis: SIMBA takes the first path for Grackle-evolved
    particles. A dust-related code path writes GIZMO-native n_e/n_H and per-H
    abundances. The dust model of Li, Narayanan & Davé 2019 is one candidate,
    but that is an inference.
  - The SIMBA module of the SIMBA team's SWIFT port shows a similar mix-up. It
    stores `e_density/rho` under the comment "all fractions are mass fraction".
    Its KIARA counterpart passes the same quantity to a function documented as
    taking n_e relative to n_H.
- *Scope.* We cannot tell whether only the output is affected or also the
  simulation's own thermal evolution. Grackle uses its normalisation
  consistently internally, so cooling of dust-free gas is probably correct. We
  do not know about the dusty path.
- *Residual non-ionized gas.* About 40% of the 4.4% of gas that remains
  non-ionized after the correction is ~$10^{4}$ K, low-density, dust-free gas.
  Its H is fully ionized, but its He is only singly ionized (EA = 0.820). This
  matches Grackle's `self_shielding_method = 3`, which switches off HeII
  photoionization. We have not confirmed which option SIMBA used.

The downstream tools are not at fault here: they implement the documented
convention.

## 5. Next steps and reproduction

- We sent our reading and correction to the SIMBA developers and the CAMELS
  team. The SIMBA team has tentatively confirmed it (section 3 and the full
  record, section 11). The CAMELS team has not replied.
- If you use EA or `NeutralHydrogenAbundance` from SIMBA or CAMELS-SIMBA, the
  check below takes about ten seconds. It uses the public CAMELS-SIMBA CV_0
  snapshot (z = 0.47, 1.75 GB, from
  `https://users.flatironinstitute.org/~camels/Sims/`):

```python
import h5py, hdf5plugin, numpy as np
with h5py.File('Sims/SIMBA/L25n256/CV/CV_0/snapshot_074.hdf5', 'r') as f:
    g = f['PartType0']
    EA, u = g['ElectronAbundance'][:], g['InternalEnergy'][:].astype(float)
    Hs = g['GrackleHI'][:] + g['GrackleHII'][:]
    dust = g['Dust_Masses'][:] > 0
    HI, HII = g['GrackleHI'][:].astype(float), g['GrackleHII'][:].astype(float)
    He = [g[k][:].astype(float) for k in ('GrackleHeI', 'GrackleHeII', 'GrackleHeIII')]

T_min = (2/3) * u * 1e10 * 1.6726e-24 * 0.588 / 1.3807e-16   # K; lower bound (mu = 0.588)
hot = T_min > 10**6.5
at088 = np.abs(EA[hot] - 0.880) < 0.005
print(at088.mean())                                # hot gas at EA = 0.880: 0.61
print(dust[hot][at088].mean())                     # ...of which dusty: 0.00
print(np.mean(np.abs(Hs[dust] - 1) < 1e-3))        # dusty gas with per-H species: 1.00

y = 0.24 / (4 * 0.76)
ne_nH = HII / (HI + HII) + y * (He[1] + 2 * He[2]) / sum(He)   # corrected n_e/n_H (SIMBA team)
ne_nH_alt = np.minimum(EA / Hs, 1 + 2 * y)                      # equivalent (section 3)
print(ne_nH.max(), np.mean(np.abs(ne_nH_alt - ne_nH) > 0.01))   # 1.157895, 1.9e-07 (3 particles)
```

The full investigation record is available on request. It includes every table,
source-code line references for GIZMO, Grackle and the SWIFT port, and the full
literature list.
