# DRAFT: message to the SIMBA and CAMELS teams (not sent)

Suggested recipients: Romeel Davé (SIMBA) and the CAMELS team
(camel.simulations@gmail.com, the contact address listed in the CAMELS docs).
Replace the bracketed placeholders before sending. Full evidence:
`simba_electron_abundance_report.md` in this folder.

---

**Subject:** Question about the `ElectronAbundance` convention in SIMBA / CAMELS-SIMBA snapshots

Dear Romeel and the CAMELS team,

I am [name, affiliation], working on kSZ/tSZ stacking with the SIMBA
m100n1024 and m50n512 runs and with CAMELS-SIMBA. While validating our
ionized-gas budgets I found that `PartType0/ElectronAbundance` in the SIMBA
snapshots seems to use two different normalisations depending on the particle,
and I would be grateful if you could confirm or correct my reading.

**What we see.** The GIZMO user guide (and IllustrisTNG) define the field as
n_e/n_H, which is how we and the standard tools (yt, the CAMELS library,
pygad, caesar, PyMSZ) read it. In SIMBA, however:

1. **Hot gas.** In m100n1024 s50 at z=0.49, about half of the gas with
   T > 3x10^6 K (a lower bound, using mu = 0.588) has EA = 0.880 exactly,
   rather than the 1.158 expected for fully ionized gas. 0.880 = X_H + Y/2 is
   the fully ionized value of n_e m_p/rho. None of these particles carries dust.
   The other half has EA = 1.158, and 99.3% of those particles carry dust. We see
   no such 0.88 population in TNG300-1 or CAMELS-TNG (0% of hot gas).

2. **Species fields.** Every gas particle falls into one of two classes:
   - GrackleHI+HII = 0.76(1-Z) and He total = 0.24 (mass fractions): EA equals
     HII + HeII/4 + HeIII/2, i.e. n_e m_p/rho, for 100% of these particles;
   - GrackleHI+HII = 1 and He total = 0.0789 = Y/4X (per H nucleus): EA equals
     HII + HeII + 2 HeIII, i.e. n_e/n_H, for 95-98% of these particles.

   The second class is exactly the set of particles with `Dust_Masses > 0`
   (P = 1.0000; P = 0.005 for dust-free gas). `NeutralHydrogenAbundance`
   equals `GrackleHI` for every particle, so it too is a mass fraction for
   dust-free gas.

3. **Reproducibility.** The same pattern appears in m100n1024 at z=0.49 and z=0,
   in all five m50n512 feedback variants, and in CAMELS-SIMBA CV_0
   (snapshots 074 and 090) and 1P_p6_2 (074). In CAMELS-SIMBA CV_0 at z=0.47,
   76% of gas particles are in the mass-fraction class.

**Why it matters.** Reading EA as n_e/n_H everywhere undercounts SIMBA's free
electrons by ~17% (m100n1024; ~18% for CAMELS-SIMBA CV_0), mostly in the diffuse
IGM, and biases temperatures derived from EA high by ~14% for the affected
ionized gas. For m100n1024 the apparent neutral gas fraction drops from 20% to
4.4% once each particle is read in its own convention. If this is right, it would
also affect published SIMBA/CAMELS-SIMBA electron-density and temperature
estimates that use the documented convention.

**Our working hypothesis.** For Grackle-evolved particles the code stores
Grackle's `e_density/density` (= n_e m_p/rho, following Grackle's convention that
e_density is n_e times the proton mass) in `SphP.Ne`, as a public pre-2019 GIZMO
fork does (`*ne_guess = my_fields.e_density[0] / my_fields.density[0]`). For
dust-bearing particles some other path writes GIZMO's native n_e/n_H and per-H
abundances. We cannot check this because the SIMBA GIZMO version is not public.
A similar pattern appears in the SIMBA module of the SWIFT port
(`src/cooling/SIMBA/cooling.c`: `e_frac = *data->e_density / rho`, with
"all fractions are mass fraction"), whose KIARA counterpart passes `e_frac` to a
function documented as taking n_e relative to n_H.

**Correction we are considering.** Per particle,
n_e/n_H = min(EA / (GrackleHI + GrackleHII), 1 + 2y), with y = (1-X_H)/(4X_H).
This reproduces n_e/n_H in both classes; only ~2x10^-7 of the gas mass needs the
cap (jet-kicked particles with EA = 1.079 but mass-fraction species).

**Questions.**
1. Is our reading of the two normalisations correct, and is the split by dust
   expected from the code?
2. Is the output field affected only, or does the mixed normalisation also enter
   the simulation's own cooling/thermal evolution?
3. Is `EA/(GrackleHI+GrackleHII)` the right way to recover n_e/n_H, or would you
   recommend another?
4. Was Grackle run with `self_shielding_method = 3`? About 40% of the gas that
   remains non-ionized after the correction is ~10^4 K, low-density, dust-free gas
   with singly ionized He.

**A one-minute check** (CAMELS-SIMBA CV_0, snapshot 074):

```python
import h5py, hdf5plugin, numpy as np
with h5py.File('Sims/SIMBA/L25n256/CV/CV_0/snapshot_074.hdf5', 'r') as f:
    g = f['PartType0']
    EA, u = g['ElectronAbundance'][:], g['InternalEnergy'][:].astype(float)
    dust = g['Dust_Masses'][:] > 0
T_min = (2/3) * u * 1e10 * 1.6726e-24 * 0.588 / 1.3807e-16      # K, lower bound
hot = T_min > 10**6.5
at088 = np.abs(EA[hot] - 0.880) < 0.005
print(at088.mean(), dust[hot][at088].mean())    # we get 0.61 and 0.00
```

I am happy to share the full set of checks. Thank you very much for your time,
and for making SIMBA and CAMELS public.

Best regards,
[name]
