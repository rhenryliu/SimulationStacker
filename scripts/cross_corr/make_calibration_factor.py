"""make_calibration_factor.py
===========================
Tasks 7, 8 and 10 of ``docs/cross_correlation_notes_v0.2_addendum.md``: the
four-amplitude calibration factor

    C_X = Y_Xm Y_gm / (Y_mm Y_gX)                                       (A12)

for the gas field ``X`` in {b, e}, in both matter conventions, for the filter
set {Sigma, DSigma, Upsilon(R0), Y-transform(Rmax)}, with jackknife errors
formed per realization.

Why this is a separate script and not a change to ``make_r_profiles.py``:
every quantity here is *post-processing* of what ``rprofiles.compute_Y_matrix``
already returns.  Three linearity facts do the work.

1. **The Park et al. (2021) Y transform is a linear combination of Sigma
   amplitudes.**  ``F^Y_R = F^Sigma_R - F^Sigma_Rmax`` as an operator on the
   map, so

       Y^(Ytr)_ab(R) = Y^(Sigma)_ab(R) - Y^(Sigma)_ab(Rmax)

   exactly, with no extra convolution -- the same trick ``compute_Y_matrix``
   already uses to build Upsilon out of DSigma.  The addendum's Appendix A
   asks for a "direct map-level filter, not a reconstruction"; this *is* the
   direct map-level filter, because the operator identity holds on the map.
   It is the ``DSigma -> Y`` reconstruction of the addendum's Section 2, which
   needs quadrature over apertures, that is a reconstruction, and that path is
   needed only for a lensing leg and is not built here.

2. **Upsilon at any R0 on the aperture grid is likewise free**, so the
   production R0 = 2' and the addendum's R0 = 1' are both carried.

3. **Convention T needs no new field sweep.**  With ``m`` = CDM and
   ``b`` = baryons, ``T = M + B`` at the mass level by construction (the CDM
   map is derived as ``total - baryon``), hence exactly

       delta_t = f_m delta_m + f_b delta_b,    f_b = <baryon>/<total>,

   and every Convention T amplitude is a bilinear recombination of amplitudes
   already measured::

       Y_gt = f_m Y_gm + f_b Y_gb
       Y_bt = f_m Y_bm + f_b Y_bb
       Y_tt = f_m^2 Y_mm + 2 f_m f_b Y_bm + f_b^2 Y_bb

   ``f_b`` is taken from the maps, never from the header: FLAMINGO carries a
   neutrino contribution inside ``Omega0`` that is absent from the particle
   maps, so ``OmegaBaryon/Omega0`` is the wrong number by a per cent.

The consequence is that one ``compute_Y_matrix`` call per (run, projection)
delivers every filter and both conventions.  The aperture grid is the Task 1
grid unioned with the reference radii that Upsilon and the Y transform need;
the nine data-matched bins over 1'-6' are untouched, because each aperture is
convolved independently.

Nothing in ``src/`` and nothing in ``make_r_profiles.py`` is modified, so
``data/r_profiles/*.npz`` remains reproducible bit-for-bit.

Usage
-----
    cd scripts/
    python cross_corr/make_calibration_factor.py \
        -p configs/cross_corr/calibration_z05.yaml
    python cross_corr/make_calibration_factor.py \
        -p configs/cross_corr/calibration_z05.yaml --sim TNG300-1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.append('../src/')
sys.path.append(str(Path(__file__).resolve().parent))

import rprofiles as rp
from stacker import SimulationStacker
from utils import comoving_to_arcmin

# Reused verbatim so the sample selection, aperture grid and labelling cannot
# drift from the Task 1 sweep.
from make_r_profiles import aperture_radii, sim_label


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

#: Gas fields whose calibration factor is computed.  'e' (ionized gas) is the
#: Gate B target of `docs/tasks_1_to_4_record.md`; 'b' (all baryons) is what
#: the suppression algebra of the note is written for.
GAS_FIELDS = ('b', 'e')

#: The gas field for which the suppression mapping (A16)/(A17) is defined.
#: Those relations descend from ``delta_t = f_m delta_m + f_b delta_b``, which
#: requires the gas field to complete the mass budget against the CDM.  The
#: electron field does not, so no suppression is written for it.
SUPPRESSION_FIELD = 'b'

#: Y-transform bins are dropped at and above this fraction of Rmax, where the
#: transform vanishes by construction.  Addendum Sections 2.5 and 8.2.
YT_USABLE_FRACTION = 0.8

#: Tolerance, in arcmin, for locating a reference radius on the aperture grid.
RADIUS_ATOL = 1e-9


def _pair(a, b):
    """Return the canonical order-independent field-pair key.

    Args:
        a (str): First field key.
        b (str): Second field key.

    Returns:
        tuple: ``(a, b)`` sorted alphabetically, matching
        ``rprofiles._pair_key``.
    """
    return (a, b) if a <= b else (b, a)


def _apply_mask(values, mask):
    """Return ``values`` with masked apertures set to NaN.

    Applies to derived quantities only (see :func:`flatten_for_npz`).  Works on
    both ``(n_rad,)`` arrays and ``(n_jk, n_rad)`` jackknife stacks.

    Args:
        values (np.ndarray): Array whose last axis runs over apertures.
        mask (np.ndarray): Boolean mask over apertures, True where informative.

    Returns:
        np.ndarray: Copy with ``~mask`` entries set to NaN, dtype float.
    """
    out = np.asarray(values, dtype=np.float64).copy()
    out[..., ~np.asarray(mask, dtype=bool)] = np.nan
    return out


def _coefficient(Y_ab, Y_aa, Y_bb):
    """Return ``Y_ab / sqrt(Y_aa Y_bb)``, NaN where the denominator is invalid.

    Mirrors ``rprofiles._coefficient`` so the coefficients reported here and
    in the Task 1 outputs are computed identically.

    Args:
        Y_ab (np.ndarray): Cross amplitude.
        Y_aa (np.ndarray): First auto amplitude.
        Y_bb (np.ndarray): Second auto amplitude.

    Returns:
        np.ndarray: Coefficient, NaN where ``Y_aa Y_bb <= 0``.
    """
    denom_sq = Y_aa * Y_bb
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom_sq > 0.0, Y_ab / np.sqrt(denom_sq), np.nan)


def _safe_divide(num, den):
    """Divide, returning NaN on a zero or invalid denominator.

    Args:
        num (np.ndarray): Numerator.
        den (np.ndarray): Denominator.

    Returns:
        np.ndarray: ``num / den``, NaN where ``den == 0`` or either is NaN.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den != 0.0, num / den, np.nan)


def radius_index(radii, target, atol=RADIUS_ATOL):
    """Locate a reference radius on the aperture grid.

    Args:
        radii (np.ndarray): Aperture grid, arcmin.
        target (float): Reference radius to locate, arcmin.
        atol (float, optional): Absolute tolerance, arcmin.  Defaults to
            :data:`RADIUS_ATOL`.

    Returns:
        int: Index of the matching aperture.

    Raises:
        ValueError: If no aperture matches.  The reference radius must be a
            grid point, because the linear-combination construction reads the
            amplitude there rather than interpolating it.
    """
    radii = np.asarray(radii, dtype=np.float64)
    hits = np.flatnonzero(np.abs(radii - float(target)) <= atol)
    if hits.size == 0:
        raise ValueError(
            f"Reference radius {target}' is not on the aperture grid "
            f"{np.array2string(radii, precision=4)}. Add it to "
            "'reference_radii' in the config: the Upsilon and Y-transform "
            "constructions read the amplitude at the reference radius and "
            "must not interpolate it."
        )
    return int(hits[0])


# ---------------------------------------------------------------------------
# Aperture grid
# ---------------------------------------------------------------------------

def calibration_radii(cfg):
    """Build the aperture grid: the Task 1 grid plus the reference radii.

    The Task 1 grid (nine data-matched bins over 1'-6' plus the diagnostic
    extension) is taken verbatim from ``make_r_profiles.aperture_radii`` and
    then unioned with every Upsilon ``R0`` and Y-transform ``Rmax`` the config
    requests.  Apertures are convolved independently, so appending reference
    radii leaves the nine data-matched bins bit-identical to the Task 1
    outputs.

    Args:
        cfg (dict): The ``stack`` config block.

    Returns:
        tuple: ``(radii, r0_list, rmax_list)`` -- the sorted aperture grid in
        arcmin, the Upsilon reference radii, and the Y-transform reference
        radii.
    """
    base = aperture_radii(cfg)
    r0_list = [float(x) for x in cfg.get('upsilon_r0', [1.0, 2.0])]
    rmax_list = [float(x) for x in cfg.get('ytransform_rmax', [4.0, 5.0, 6.0, 9.0])]

    grid = np.concatenate([base, np.asarray(r0_list + rmax_list, dtype=float)])
    grid = np.unique(np.round(grid, 9))
    return grid, r0_list, rmax_list


# ---------------------------------------------------------------------------
# Field loading
# ---------------------------------------------------------------------------

def load_fields_and_baryon_fraction(stacker, n_pixels, projection, cfg,
                                    verbose=True):
    """Load the projected fields, derive CDM, and measure the baryon fraction.

    This mirrors ``make_r_profiles.load_component_fields`` but additionally
    returns ``f_b = <baryon>/<total>`` measured on the maps themselves, which
    Convention T needs.  The map-derived value is the correct one: the
    identity ``delta_t = f_m delta_m + f_b delta_b`` holds exactly for
    ``f_b = <baryon>/<total>`` because the CDM map is *defined* as
    ``total - baryon``, whereas ``OmegaBaryon/Omega0`` from the header differs
    at the per-cent level for suites with massive neutrinos.

    Args:
        stacker (SimulationStacker): Configured stacker.
        n_pixels (int): Pixels per side of the cached grid.
        projection (str): 'xy', 'xz' or 'yz'.
        cfg (dict): The ``stack`` config block.
        verbose (bool, optional): Print progress.  Defaults to True.

    Returns:
        tuple: ``(deltas, f_b)`` with ``deltas`` keyed 'e', 'b', 'm' and
        ``f_b`` the map-derived baryon mass fraction.
    """
    load = cfg.get('load_field', True)
    save = cfg.get('save_field', False)
    ptype_e = cfg.get('particle_type_e', 'ionized_gas')
    ptype_b = cfg.get('particle_type_b', 'baryon')

    def _field(ptype):
        if verbose:
            print(f'    loading {ptype} ...', flush=True)
        return stacker.makeField(ptype, nPixels=n_pixels,
                                 projection=projection, save=save, load=load)

    deltas = {}

    baryon = _field(ptype_b)
    total = _field('total')

    mean_b = float(np.mean(baryon, dtype=np.float64))
    mean_t = float(np.mean(total, dtype=np.float64))
    f_b = mean_b / mean_t

    cdm = rp.derive_cdm_field(total, baryon, header=stacker.header,
                              verbose=verbose)
    del total
    deltas['m'] = rp.to_overdensity(cdm)
    del cdm
    deltas['b'] = rp.to_overdensity(baryon)
    del baryon

    electrons = _field(ptype_e)
    deltas['e'] = rp.to_overdensity(electrons)
    del electrons

    if verbose:
        f_header = (1.0 - float(stacker.header['OmegaBaryon'])
                    / float(stacker.header['Omega0'])) if 'OmegaBaryon' in stacker.header else np.nan
        print(f'    baryon mass fraction f_b = {f_b:.6f} (maps); '
              f'1 - f_b from header cosmology = {f_header:.6f}')

    return deltas, f_b


# ---------------------------------------------------------------------------
# Filter assembly
# ---------------------------------------------------------------------------

def assemble_derived_filter(Y_base, Y_base_jk, radii, kind, ref):
    """Assemble Upsilon or the Y transform from a base filter's amplitudes.

    Both derived filters are *linear combinations of the base filter at two
    apertures*, so they are formed at the amplitude level rather than by extra
    convolutions, and they are formed per jackknife realization so that the
    correlations between apertures survive into the errors:

    - ``Upsilon(R; R0) = DSigma(R) - (R0/R)^2 DSigma(R0)``  (Baldauf+ 2010)
    - ``Y(R; Rmax)     = Sigma(R) - Sigma(Rmax)``           (Park+ 2021)

    Args:
        Y_base (dict): ``{pair: (n_rad,)}`` full-map base amplitudes.
        Y_base_jk (dict): ``{pair: (n_jk, n_rad)}`` leave-one-out amplitudes.
        radii (np.ndarray): Aperture grid, arcmin.
        kind (str): ``'Upsilon'`` or ``'Ytransform'``.
        ref (float): ``R0`` for Upsilon, ``Rmax`` for the Y transform, arcmin.

    Returns:
        tuple: ``(Y, Y_jk, mask)`` with the same dict layout as the inputs and
        a boolean mask over ``radii`` marking the informative bins.

    Raises:
        ValueError: If ``kind`` is unknown, or ``ref`` is not on the grid.
    """
    idx = radius_index(radii, ref)
    radii = np.asarray(radii, dtype=np.float64)

    if kind == 'Upsilon':
        # (R0/R)^2 reference term: nulls everything below R0.
        scale = (float(ref) / radii) ** 2
        mask = rp.upsilon_defined_mask(radii, ref)
    elif kind == 'Ytransform':
        scale = np.ones_like(radii)
        # Y(Rmax) == 0 by construction and the bins just below it carry almost
        # no signal, so they are dropped exactly as R == R0 is for Upsilon.
        mask = radii < YT_USABLE_FRACTION * float(ref)
    else:
        raise ValueError(
            f"kind must be 'Upsilon' or 'Ytransform', got {kind!r}.")

    Y = {}
    Y_jk = {}
    for pair, values in Y_base.items():
        Y[pair] = values - scale * values[idx]
        jk = Y_base_jk[pair]
        Y_jk[pair] = jk - scale[None, :] * jk[:, idx][:, None]
    return Y, Y_jk, mask


def add_convention_t(Y, f_b):
    """Add the total-matter (Convention T) amplitudes in place.

    ``delta_t = f_m delta_m + f_b delta_b`` holds exactly on these maps (the
    CDM map is derived as ``total - baryon``), and every filter here is
    linear, so the total-matter amplitudes are exact bilinear recombinations
    of amplitudes already measured.  No new field and no new convolution.

    Works unchanged on full-map arrays ``(n_rad,)`` and on jackknife stacks
    ``(n_jk, n_rad)``, since the algebra is elementwise.

    Only the cross terms whose inputs are present are built, so a dict
    carrying just the matter and baryon autos still yields ``Y_tt`` -- which
    is all :func:`term_b_diagnostic` needs.

    Args:
        Y (dict): ``{pair: array}`` amplitudes for one filter.  Must contain
            at least the ``mm``, ``bm`` and ``bb`` pairs.  Modified in place.
        f_b (float): Baryon mass fraction ``<baryon>/<total>``.

    Returns:
        dict: The same dict, with the available ``'t'`` pairs added.

    Raises:
        KeyError: If the ``mm``, ``bm`` or ``bb`` pair is missing; those three
            define the total-matter field and have no fallback.
    """
    fm = 1.0 - float(f_b)
    fb = float(f_b)

    Y_mm = Y[_pair('m', 'm')]
    Y_bm = Y[_pair('b', 'm')]
    Y_bb = Y[_pair('b', 'b')]

    Y[_pair('b', 't')] = fm * Y_bm + fb * Y_bb
    Y[_pair('m', 't')] = fm * Y_mm + fb * Y_bm
    Y[_pair('t', 't')] = fm * fm * Y_mm + 2.0 * fm * fb * Y_bm + fb * fb * Y_bb

    # The tracer cross terms exist only when that tracer was measured.
    for tracer in ('g', 'e'):
        with_m = _pair(tracer, 'm')
        with_b = _pair(tracer, 'b')
        if with_m in Y and with_b in Y:
            Y[_pair(tracer, 't')] = fm * Y[with_m] + fb * Y[with_b]
    return Y


def build_filters(Ymat, radii, r0_list, rmax_list, f_b):
    """Build every filter's amplitudes, in both matter conventions.

    Args:
        Ymat (dict): Output of ``rprofiles.compute_Y_matrix``.
        radii (np.ndarray): Aperture grid, arcmin.
        r0_list (sequence): Upsilon reference radii, arcmin.
        rmax_list (sequence): Y-transform reference radii, arcmin.
        f_b (float): Map-derived baryon mass fraction.

    Returns:
        dict: ``{filter_name: {'Y': {pair: (n_rad,)},
        'Y_jk': {pair: (n_jk, n_rad)}, 'mask': (n_rad,) bool}}``.
    """
    n_rad = len(radii)
    out = {}

    for base in ('Sigma', 'DSigma'):
        out[base] = {
            'Y': dict(Ymat['Y'][base]),
            'Y_jk': dict(Ymat['Y_jk'][base]),
            'mask': np.ones(n_rad, dtype=bool),
        }

    for r0 in r0_list:
        Y, Y_jk, mask = assemble_derived_filter(
            Ymat['Y']['DSigma'], Ymat['Y_jk']['DSigma'], radii, 'Upsilon', r0)
        out[f'Upsilon_R0={r0:g}'] = {'Y': Y, 'Y_jk': Y_jk, 'mask': mask}

    for rmax in rmax_list:
        Y, Y_jk, mask = assemble_derived_filter(
            Ymat['Y']['Sigma'], Ymat['Y_jk']['Sigma'], radii, 'Ytransform',
            rmax)
        out[f'Ytransform_Rmax={rmax:g}'] = {'Y': Y, 'Y_jk': Y_jk, 'mask': mask}

    for entry in out.values():
        add_convention_t(entry['Y'], f_b)
        add_convention_t(entry['Y_jk'], f_b)

    return out


# ---------------------------------------------------------------------------
# Calibration factors
# ---------------------------------------------------------------------------

def calibration_factors(Y, gas='b'):
    """Form the calibration factors and suppression variables for one filter.

    Every quantity is a pure function of the filtered amplitudes, so calling
    this on a full-map amplitude dict and on a jackknife stack gives the
    per-realization values the addendum's Appendix A trap 5 demands.

    Quantities, for gas field ``X`` and matter field ``m`` (CDM, Convention C)
    or ``t`` (total, Convention T):

    - ``C`` = ``Y_Xm Y_gm / (Y_mm Y_gX)``, the Route B factor (A12).  Equals
      unity identically when the galaxy-gas correlation is entirely mediated
      by the matter field.
    - ``C_A`` = ``r_Xm / r_gX`` = ``Y_Xm sqrt(Y_gg) / (Y_gX sqrt(Y_mm))``, the
      Route A factor (A12).  ``Y_XX`` cancels out of it algebraically.
    - ``x`` = ``Y_Xm / Y_mm``, the filtered analogue of ``P_bm/P_mm``.
    - ``r_gX``, ``r_Xm``, ``r_gm``: the individual coefficients, diagnostics
      only (they are not bounded by unity for compensated filters).

    Args:
        Y (dict): ``{pair: array}`` amplitudes for one filter, including the
            Convention T pairs.
        gas (str, optional): Gas field key, 'b' or 'e'.  Defaults to 'b'.

    Returns:
        dict: Named arrays, each the shape of the input amplitude arrays.
    """
    X = gas
    Y_gg = Y[_pair('g', 'g')]
    Y_XX = Y[_pair(X, X)]
    Y_gX = Y[_pair('g', X)]

    out = {}
    for tag, mkey in (('', 'm'), ('_t', 't')):
        Y_mm = Y[_pair(mkey, mkey)]
        Y_Xm = Y[_pair(X, mkey)]
        Y_gm = Y[_pair('g', mkey)]

        out[f'C{tag}'] = _safe_divide(Y_Xm * Y_gm, Y_mm * Y_gX)
        with np.errstate(invalid='ignore'):
            root = np.where(Y_gg > 0.0, np.sqrt(Y_gg), np.nan)
            root_m = np.where(Y_mm > 0.0, np.sqrt(Y_mm), np.nan)
        out[f'CA{tag}'] = _safe_divide(Y_Xm * root, Y_gX * root_m)
        out[f'x{tag}'] = _safe_divide(Y_Xm, Y_mm)
        out[f'r_gm{tag}'] = _coefficient(Y_gm, Y_gg, Y_mm)
        out[f'r_Xm{tag}'] = _coefficient(Y_Xm, Y_XX, Y_mm)

    out['r_gX'] = _coefficient(Y_gX, Y_gg, Y_XX)
    return out


def suppression(x, f_b, convention='C'):
    """Map the filtered cross-correlation ratio to the power suppression.

    Addendum Eqs. (A16) and (A17):

    - Convention C: ``S = (f_m + f_b x)^2``
    - Convention T: ``S = f_m^2 / (1 - f_b x_t)^2``

    Both adopt ``P_mm^hydro = P_tt^DMO``, i.e. they neglect the 1-2 per cent
    hydrodynamic back-reaction on the CDM, which is booked separately.

    Args:
        x (np.ndarray): ``Y_bm/Y_mm`` (Convention C) or ``Y_bt/Y_tt``
            (Convention T).
        f_b (float): Baryon mass fraction.
        convention (str, optional): ``'C'`` or ``'T'``.  Defaults to ``'C'``.

    Returns:
        np.ndarray: The suppression ``S``.

    Raises:
        ValueError: If ``convention`` is not 'C' or 'T'.
    """
    fb = float(f_b)
    fm = 1.0 - fb
    if convention == 'C':
        return (fm + fb * x) ** 2
    if convention == 'T':
        return _safe_divide(fm ** 2 * np.ones_like(x), (1.0 - fb * x) ** 2)
    raise ValueError(f"convention must be 'C' or 'T', got {convention!r}.")


def term_b_diagnostic(Y, f_b):
    """Test the addendum Section 4.6 claim about Stage 8's 'term B'.

    ``check_theory_transfer.py`` reports ``B = Y_tt/Y_mm - 1`` measured on the
    same hydrodynamic box and books it as the dominant *theory-side
    systematic*.  Addendum Section 4.6 argues instead that it is the signal,
    because Eq. (A15) gives

        Y_tt/Y_mm = (f_m + f_b x)^2 + f_b^2 (1 - r_bm^2) Y_bb/Y_mm,

    whose second term the addendum bounds below 1e-3.  This returns both sides
    so the claim is checked rather than asserted.

    Args:
        Y (dict): Full-map amplitudes for one filter, with Convention T added.
        f_b (float): Baryon mass fraction.

    Returns:
        dict: ``'B_measured'``, ``'B_predicted'`` and their difference
        ``'B_residual'``, the last being the neglected stochastic term.
    """
    x = _safe_divide(Y[_pair('b', 'm')], Y[_pair('m', 'm')])
    measured = _safe_divide(Y[_pair('t', 't')], Y[_pair('m', 'm')]) - 1.0
    predicted = suppression(x, f_b, 'C') - 1.0
    return {'B_measured': measured,
            'B_predicted': predicted,
            'B_residual': measured - predicted}


# ---------------------------------------------------------------------------
# Per-simulation driver
# ---------------------------------------------------------------------------

def process_simulation(sim_type, sim_entry, cfg, out_dir, verbose=True):
    """Compute and save the calibration factors for one simulation.

    Args:
        sim_type (str): Simulation suite.
        sim_entry (dict): One entry of the config ``sims`` list.
        cfg (dict): The ``stack`` config block.
        out_dir (pathlib.Path): Directory for the output ``.npz`` files.
        verbose (bool, optional): Print progress.  Defaults to True.

    Returns:
        list: Paths of the written ``.npz`` files.
    """
    name = sim_entry['name']
    snapshot = sim_entry['snapshot']
    feedback = sim_entry.get('feedback')
    n_pixels = sim_entry['n_pixels']
    label = sim_label(sim_type, name, feedback)

    z_cfg = float(sim_entry.get('redshift', cfg.get('redshift', 0.5)))
    stacker = SimulationStacker(name, snapshot, nPixels=n_pixels,
                                simType=sim_type, feedback=feedback, z=z_cfg)

    z_true = float(np.asarray(stacker.header['Redshift']).ravel()[0])
    lbox = float(stacker.header['BoxSize'])
    theta_arcmin = comoving_to_arcmin(lbox, z_true, cosmo=stacker.cosmo)
    pixel_arcmin = theta_arcmin / n_pixels

    radii, r0_list, rmax_list = calibration_radii(cfg)
    if not r0_list:
        raise ValueError(
            "'upsilon_r0' is empty; at least one Upsilon reference radius is "
            "required (compute_Y_matrix needs one internally).")
    dr = float(cfg.get('dr_arcmin', rp.DR_ARCMIN))
    n_jk_side = int(cfg.get('n_jk_side', rp.N_JK_SIDE))
    target = float(cfg.get('halo_abundance_target', 5e-4))
    parent_upper = cfg.get('parent_mass_upper', 5e14)
    parent_upper = None if parent_upper is None else float(parent_upper)

    if verbose:
        print(f'\n{"=" * 70}')
        print(f'{label}  ({sim_type}, snapshot {snapshot})')
        print(f'{"=" * 70}')
        print(f'  z (header) = {z_true:.4f}   [config {z_cfg}]')
        print(f'  box = {lbox / 1000:.1f} cMpc/h = {theta_arcmin:.1f} arcmin')
        print(f'  grid = {n_pixels}^2, pixel = {pixel_arcmin:.5f} arcmin')
        print(f'  {len(radii)} apertures {radii.min():.3f}-{radii.max():.3f} '
              f'arcmin')
        print(f"  Upsilon R0 = {r0_list}, Y-transform Rmax = {rmax_list}")

    # Resolution guard: every aperture, including the reference radii, must be
    # resolved before any field is loaded.
    for guard in radii:
        rp.build_aperture_kernel(n_pixels, pixel_arcmin, float(guard),
                                 'DSigma', dr)

    subhalos = stacker.loadSubHalos()

    written = []
    for projection in cfg.get('projections', ['yz']):
        t0 = time.time()
        if verbose:
            print(f'\n  --- projection {projection} ---', flush=True)

        deltas, f_b = load_fields_and_baryon_fraction(
            stacker, n_pixels, projection, cfg, verbose=verbose)

        galaxies, halo_mask = rp.make_galaxy_field(
            stacker, projection, n_pixels, target,
            parent_mass_upper=parent_upper, subhalos=subhalos)
        n_gal = int(halo_mask.size)
        nbar_pix = galaxies.sum() / float(n_pixels * n_pixels)
        deltas['g'] = rp.to_overdensity(galaxies)
        del galaxies
        if verbose:
            print(f'    {n_gal} SHAM galaxies (nbar = {nbar_pix:.4e} / pixel)',
                  flush=True)
            print('    computing filtered amplitudes ...', flush=True)

        # r0 is irrelevant here -- Upsilon is rebuilt below for every R0 in
        # the config -- but compute_Y_matrix requires one, and passing a grid
        # point avoids appending a redundant aperture.
        Ymat = rp.compute_Y_matrix(deltas, pixel_arcmin, radii=radii, dr=dr,
                                   r0=float(r0_list[0]), nbar_pix=nbar_pix,
                                   n_jk_side=n_jk_side)
        del deltas

        filters = build_filters(Ymat, radii, r0_list, rmax_list, f_b)

        meta = {
            'label': label,
            'sim_type': sim_type,
            'sim_name': str(name),
            'feedback': str(feedback),
            'snapshot': snapshot,
            'projection': projection,
            'redshift': z_true,
            'n_pixels': n_pixels,
            'pixel_arcmin': pixel_arcmin,
            'boxsize_ckpc_h': lbox,
            'n_galaxies': n_gal,
            'nbar_pix': nbar_pix,
            'abundance_target': target,
            'dr_arcmin': dr,
            'f_b': f_b,
            'n_jk': Ymat['n_jk'],
            'filters': np.array(sorted(filters), dtype=object),
            'gas_fields': np.array(GAS_FIELDS, dtype=object),
        }

        payload = flatten_for_npz(filters, radii, f_b, meta)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir /
                    f'calibration_{label}_{snapshot}_{projection}.npz')
        np.savez(out_path, **payload)
        written.append(out_path)

        if verbose:
            print(f'    saved {out_path}  ({time.time() - t0:.1f} s)')
            report_run(filters, radii, f_b)

    return written


def flatten_for_npz(filters, radii, f_b, meta):
    """Flatten the nested filter results into a flat ``np.savez`` payload.

    Args:
        filters (dict): Output of :func:`build_filters`.
        radii (np.ndarray): Aperture grid, arcmin.
        f_b (float): Baryon mass fraction.
        meta (dict): Scalar metadata.

    Masking convention, which a reader of the ``.npz`` must know:

    - the raw amplitudes ``Y_*`` and ``Yjk_*`` are written **unmasked**, since
      they are well-defined at every aperture and are useful as diagnostics
      (``Upsilon`` below ``R0`` is a real number, just not the estimator);
    - every *derived* quantity -- ``C``, ``CA``, ``x``, the coefficients and
      ``S`` -- is written with masked bins set to NaN, because outside the
      mask those are not the quantity they are named after. Without this a
      consumer that forgot to intersect with ``mask_<filter>`` would silently
      plot, for instance, ``C = 12.5`` at ``R < R0`` for ``Upsilon``.

    Returns:
        dict: Flat mapping of names to arrays.
    """
    out = {'radii': radii}

    for fname, entry in filters.items():
        mask = entry['mask']
        out[f'mask_{fname}'] = mask
        for (a, b), values in entry['Y'].items():
            out[f'Y_{a}{b}_{fname}'] = values
            out[f'Yjk_{a}{b}_{fname}'] = entry['Y_jk'][(a, b)]

        for gas in GAS_FIELDS:
            full = calibration_factors(entry['Y'], gas=gas)
            jk = calibration_factors(entry['Y_jk'], gas=gas)
            for key, values in full.items():
                out[f'{key}_{gas}_{fname}'] = _apply_mask(values, mask)
                out[f'{key}jk_{gas}_{fname}'] = _apply_mask(jk[key], mask)
                out[f'{key}err_{gas}_{fname}'] = _apply_mask(
                    rp.jackknife_error(jk[key], axis=0), mask)

            # Suppression is defined ONLY for the total baryon field.  Eqs.
            # (A16)/(A17) descend from delta_t = f_m delta_m + f_b delta_b,
            # in which 'b' completes the mass budget against the CDM.
            # Electrons are a strict subset of the baryons, not a
            # complementary component, so there is no analogous identity and
            # S(x_e) would be a finite, plausible-looking number that means
            # nothing.  C, C_A, x and the coefficients ARE meaningful for the
            # electron field -- it is the Gate B target -- and are written
            # above for both.
            if gas != SUPPRESSION_FIELD:
                continue
            for tag, conv in (('', 'C'), ('_t', 'T')):
                s_full = suppression(full[f'x{tag}'], f_b, conv)
                s_jk = suppression(jk[f'x{tag}'], f_b, conv)
                out[f'S{tag}_{gas}_{fname}'] = _apply_mask(s_full, mask)
                out[f'S{tag}jk_{gas}_{fname}'] = _apply_mask(s_jk, mask)
                out[f'S{tag}err_{gas}_{fname}'] = _apply_mask(
                    rp.jackknife_error(s_jk, axis=0), mask)

        for key, values in term_b_diagnostic(entry['Y'], f_b).items():
            out[f'{key}_{fname}'] = values

    for key, value in meta.items():
        out[f'meta_{key}'] = np.array(value)

    return out


def report_run(filters, radii, f_b):
    """Print a compact per-filter summary for one run.

    Args:
        filters (dict): Output of :func:`build_filters`.
        radii (np.ndarray): Aperture grid, arcmin.
        f_b (float): Baryon mass fraction.
    """
    # Each filter masks a different set of bins -- Upsilon drops R <= R0, the
    # Y transform drops R >= 0.8 Rmax -- so the reported endpoints are the
    # first and last *usable* aperture and are labelled with their radii
    # rather than assumed to be 1' and 6'.
    data = (radii >= 1.0) & (radii <= 6.0)
    print(f"      {'filter':24s} {'|C-1| max':>10s} {'R_lo':>6s} "
          f"{'C(R_lo)':>9s} {'R_hi':>6s} {'C(R_hi)':>9s} {'CA(R_lo)':>10s} "
          f"{'B resid':>10s}")
    for fname in sorted(filters):
        entry = filters[fname]
        use = data & entry['mask']
        if not use.any():
            continue
        fac = calibration_factors(entry['Y'], gas='b')
        C = fac['C']
        resid = term_b_diagnostic(entry['Y'], f_b)['B_residual']
        first = np.flatnonzero(use)[0]
        last = np.flatnonzero(use)[-1]
        with np.errstate(invalid='ignore'):
            worst = np.nanmax(np.abs(C[use] - 1.0))
        print(f'      {fname:24s} {worst:10.4f} {radii[first]:6.3f} '
              f'{C[first]:9.4f} {radii[last]:6.3f} {C[last]:9.4f} '
              f'{fac["CA"][first]:10.4f} '
              f'{np.nanmax(np.abs(resid[use])):10.2e}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(path2config, sim=None, feedback=None, verbose=True):
    """Run the calibration-factor computation for every simulation in a config.

    Args:
        path2config (str): Path to the YAML configuration file.
        sim (str, optional): Only process simulations with this name.
        feedback (str, optional): Only process this feedback variant.
        verbose (bool, optional): Print progress.  Defaults to True.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)

    cfg = config.get('stack', {})
    out_dir = Path(config.get('plot', {}).get('npz_path',
                                              '../data/cross_corr_C/'))

    t_start = time.time()
    written = []
    for suite in config['simulations']:
        sim_type = suite['sim_type']
        for entry in suite['sims']:
            if sim is not None and entry['name'] != sim:
                continue
            if feedback is not None and entry.get('feedback') != feedback:
                continue
            written.extend(
                process_simulation(sim_type, entry, cfg, out_dir,
                                   verbose=verbose))

    print(f'\n{"=" * 70}')
    print(f'Wrote {len(written)} file(s) in '
          f'{(time.time() - t_start) / 60:.1f} minutes:')
    for path in written:
        print(f'  {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute the addendum calibration factor C from cached '
                    'projected fields.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/calibration_z05.yaml',
                        help='Path to the YAML configuration file.')
    parser.add_argument('--sim', type=str, default=None,
                        help='Only process this simulation name.')
    parser.add_argument('--feedback', type=str, default=None,
                        help='Only process this feedback variant.')
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
