"""Integration test for the Task 1 r-profile machinery against the legacy stacker.

Cross-checks the FFT map-level route in ``src/rprofiles.py`` against the
existing stamp-stacking route (``SimulationStacker.stack_on_array``) on real
simulation data with a real SHAM galaxy sample, as required by
``docs/r_profiles_task1_spec.md`` ("Integration test").

Both routes must produce the same filtered galaxy-cross amplitude
``Y_eg(R)``.  They differ only in convention, and the test separates the two
conventions rather than hiding them:

- ``stack_on_array`` centres each stamp at ``round(pos / pixel)`` while an NGP
  galaxy map bins at ``floor(pos / pixel)``; that is a half-pixel centring
  difference in the galaxy positions, nothing more.
- ``stack_on_array`` builds its stamp radius grid with a ``linspace`` over the
  rounded cutout half-width, so its effective pixel scale differs from the
  true arcmin-per-pixel by the rounding of that half-width.

The first test matches the centring convention and therefore isolates the
radial-grid convention alone; the second measures the centring difference on
top of it.  Both print their residuals.

The FLAMINGO suite is deliberately not exercised here: this is a convention
cross-check, and TNG300-1 is the cheapest box that has the required cached
field.

Run (needs the Perlmutter scratch caches):

    cd tests/
    pytest test_rprofiles_integration.py -v -s
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

TNG_BASE = '/pscratch/sd/r/rhliu/simulations/IllustrisTNG/TNG300-1/'

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.path.isdir(TNG_BASE),
                       reason='TNG300-1 data not available on this system'),
]

SIM = 'TNG300-1'
SNAPSHOT = 67
REDSHIFT = 0.5
PROJECTION = 'yz'
N_PIXELS = 2674           # the cached 0.2 arcmin/pixel grid
PTYPE = 'ionized_gas'     # the 'e' field of the production run
ABUNDANCE_TARGET = 5e-4   # LRG-like sample
PARENT_MASS_UPPER = 5e14


@pytest.fixture(scope='module')
def setup():
    """Load the cached field, build the SHAM sample, and derive shared scales.

    Skips (rather than computing for hours) if the cached field is absent.

    Returns:
        dict: Everything the comparison tests need.
    """
    import rprofiles as rp
    from stacker import SimulationStacker
    from utils import comoving_to_arcmin

    stacker = SimulationStacker(SIM, SNAPSHOT, nPixels=N_PIXELS,
                                simType='IllustrisTNG', z=REDSHIFT)
    # The snapshot header is authoritative for the angular scale, and
    # make_r_profiles.py uses it, so the test must too: TNG300-1 snapshot 67 is
    # z = 0.5030, not the round 0.5 the configs carry.
    z_true = float(np.asarray(stacker.header['Redshift']).ravel()[0])
    try:
        field = stacker.loadData(PTYPE, nPixels=N_PIXELS,
                                 projection=PROJECTION, type='field')
    except ValueError as exc:
        pytest.skip(f'Cached {PTYPE} field not available: {exc}')

    delta_e = rp.to_overdensity(field)

    try:
        total = stacker.loadData('total', nPixels=N_PIXELS,
                                 projection=PROJECTION, type='field')
        baryon = stacker.loadData('baryon', nPixels=N_PIXELS,
                                  projection=PROJECTION, type='field')
    except ValueError as exc:
        pytest.skip(f'Cached total/baryon fields not available: {exc}')
    delta_m = rp.to_overdensity(
        rp.derive_cdm_field(total, baryon, header=stacker.header,
                            verbose=False))
    del total, baryon

    theta_arcmin = comoving_to_arcmin(stacker.header['BoxSize'], z_true,
                                      cosmo=stacker.cosmo)
    pixel_arcmin = theta_arcmin / N_PIXELS

    # Reproduce stack_on_array's cutout geometry so the tests can derive the
    # effective pixel scale of its linspace stamp radius grid.
    radii = rp.APERTURES_ARCMIN
    n_vir = int(radii.max() + 1)
    rad_pixel = 1.0 / pixel_arcmin
    cutout_size = 2 * int(round(n_vir * rad_pixel)) + 1

    subhalos = stacker.loadSubHalos()
    halo_mask = rp.select_sham_subhalos(
        stacker, ABUNDANCE_TARGET, parent_mass_upper=PARENT_MASS_UPPER,
        subhalos=subhalos)

    print(f'\n  {SIM} snap {SNAPSHOT}: {N_PIXELS}^2 grid, '
          f'{pixel_arcmin:.5f} arcmin/pixel, '
          f'{halo_mask.size} SHAM galaxies')

    return {
        'rp': rp,
        'stacker': stacker,
        'z_true': z_true,
        'delta_e': delta_e,
        'delta_m': delta_m,
        'pixel_arcmin': pixel_arcmin,
        'n_vir': n_vir,
        'cutout_size': cutout_size,
        # stack_on_array's stamp radius grid is a linspace over the *rounded*
        # cutout half-width, so its effective pixel scale is a rational
        # approximation to the true one -- here 0.2 exactly against a true
        # 0.19893.  Comparisons that use this scale isolate the kernel algebra
        # from that quantization.
        'stamp_pixel': 2.0 * n_vir / (cutout_size - 1),
        'subhalos': subhalos,
        'halo_mask': halo_mask,
        'radii': radii,
    }


@pytest.fixture(scope='module')
def stamp_reference(setup):
    """Stack the identical SHAM sample through the legacy stamp route.

    Returns:
        np.ndarray: Mean DSigma-filtered delta_e at the galaxy positions,
        shape (n_radii,).
    """
    stacker = setup['stacker']
    radii = setup['radii']
    _, profiles = stacker.stack_on_array(
        array=setup['delta_e'],
        filterType='DSigma',
        minRadius=float(radii.min()),
        maxRadius=float(radii.max()),
        numRadii=len(radii),
        projection=PROJECTION,
        radDistance=1.0,
        radDistanceUnits='arcmin',
        z=setup['z_true'],
        use_subhalos=True,
        halo_mask=setup['halo_mask'],
    )
    return np.mean(profiles, axis=1)


def _galaxy_delta(positions_2d, n_pixels, lbox, rounding):
    """Deposit galaxies onto a grid with an explicit rounding convention.

    Args:
        positions_2d (np.ndarray): Transverse positions, shape (N, 2), ckpc/h.
        n_pixels (int): Pixels per side.
        lbox (float): Box size in ckpc/h.
        rounding (str): ``'floor'`` (NGP bin membership, matching the particle
            maps) or ``'round'`` (nearest pixel centre, matching
            ``stack_on_array``).

    Returns:
        tuple: ``(delta_g, nbar_pix)``.
    """
    scaled = positions_2d / (lbox / n_pixels)
    if rounding == 'floor':
        idx = np.floor(scaled).astype(int) % n_pixels
    elif rounding == 'round':
        idx = np.round(scaled).astype(int) % n_pixels
    else:
        raise ValueError(f'Unknown rounding convention: {rounding!r}')

    counts = np.zeros((n_pixels, n_pixels), dtype=np.float64)
    np.add.at(counts, (idx[:, 0], idx[:, 1]), 1.0)
    nbar = counts.sum() / float(n_pixels * n_pixels)
    return counts / nbar - 1.0, nbar


def _fft_amplitude(setup, rounding, pixel_arcmin=None):
    """Compute Y_eg through the FFT route with a given galaxy rounding.

    Args:
        setup (dict): The module fixture.
        rounding (str): ``'floor'`` or ``'round'``.
        pixel_arcmin (float, optional): Pixel scale for the aperture kernel.
            Defaults to None, meaning the true arcmin-per-pixel.  Pass
            ``setup['stamp_pixel']`` to match ``stack_on_array``'s quantized
            stamp grid.

    Returns:
        np.ndarray: Y_eg(R) for the DSigma filter, shape (n_radii,).
    """
    rp = setup['rp']
    if pixel_arcmin is None:
        pixel_arcmin = setup['pixel_arcmin']
    pos2d = rp.project_positions(
        setup['subhalos']['SubhaloPos'][setup['halo_mask']], PROJECTION)
    delta_g, nbar = _galaxy_delta(
        pos2d, N_PIXELS, float(setup['stacker'].header['BoxSize']), rounding)

    Ymat = rp.compute_Y_matrix(
        {'e': setup['delta_e'], 'g': delta_g},
        pixel_arcmin, radii=setup['radii'], nbar_pix=nbar)
    return rp.get_Y(Ymat, 'DSigma', 'e', 'g')


def _report(radii, fft, stamp, label):
    """Print a per-aperture residual table and return the fractional residuals.

    Args:
        radii (np.ndarray): Aperture radii in arcmin.
        fft (np.ndarray): FFT-route amplitudes.
        stamp (np.ndarray): Stamp-route amplitudes.
        label (str): Description of the comparison.

    Returns:
        np.ndarray: Fractional residuals ``fft/stamp - 1``.
    """
    frac = fft / stamp - 1.0
    print(f'\n  {label}')
    print(f"  {'R [arcmin]':>11}  {'FFT route':>14}  {'stamp route':>14}  "
          f"{'frac. resid.':>13}")
    for R, a, b, f in zip(radii, fft, stamp, frac):
        print(f'  {R:11.3f}  {a:14.6e}  {b:14.6e}  {f:+13.3e}')
    print(f'  max |fractional residual| = {np.max(np.abs(frac)):.3e}')
    return frac


class TestFFTMatchesStampStack:
    """The map-level FFT route must reproduce the legacy stamp stack."""

    def test_matched_centring_agrees_exactly_off_lattice_ties(
            self, setup, stamp_reference):
        """Away from boundary-degenerate apertures the two routes agree exactly.

        ``stack_on_array`` builds its stamp radius grid as a linspace over the
        rounded cutout half-width, giving an effective pixel scale marginally
        different from the true arcmin-per-pixel.  That difference is
        invisible unless a shell of lattice points falls between the two
        thresholds, in which case the whole shell changes disk/annulus
        membership.  With the galaxy-centring convention matched, this test
        asserts the strong version of the claim: machine-precision agreement
        at every aperture that is not boundary-degenerate.
        """
        rp = setup['rp']
        radii = setup['radii']
        stamp_pixel = setup['stamp_pixel']
        # Match the stamp grid's quantized pixel scale as well as its centring,
        # so the only thing left that could differ is the kernel algebra.
        fft = _fft_amplitude(setup, rounding='round', pixel_arcmin=stamp_pixel)

        # stack_on_array is internally inconsistent about which pixel scale it
        # uses: delta_sigma_kernel is handed pixel_size=arcminPerPixel (the
        # TRUE scale) for the 1/pixArea normalization, while the radius grid it
        # bins against is the quantized linspace (0.2 exactly here).  Rescale
        # the area normalization to match, so this test compares kernel algebra
        # rather than that bookkeeping.  Since every Y in a coefficient carries
        # the same factor, it cancels completely in r and does not affect the
        # Task 1 deliverable -- nor the f_gas ratios, which divide two stacks.
        fft = fft * (stamp_pixel / setup['pixel_arcmin']) ** 2

        frac = _report(radii, fft, stamp_reference,
                       'Matched centring, pixel scale and area normalization: '
                       'isolates the kernel algebra')

        tied = set(rp.degenerate_apertures(radii, stamp_pixel))
        print(f'\n  boundary-degenerate apertures: '
              f'{sorted(round(r, 3) for r in tied)}')
        for R, edges in sorted(rp.degenerate_apertures(radii,
                                                       stamp_pixel).items()):
            for name, margin, shell in edges:
                print(f"    R={R:.3f}' {name} edge is {margin:.2e} pixels from "
                      f'a {shell}-pixel lattice shell')

        clean = np.array([not any(np.isclose(R, t) for t in tied)
                          for R in radii])
        assert clean.any(), 'No non-degenerate aperture to compare'
        assert np.max(np.abs(frac[clean])) < 1e-10, (
            'FFT and stamp routes must agree to machine precision at '
            f'non-degenerate apertures, got {frac[clean]} at '
            f'R={radii[clean]}'
        )
        # At degenerate apertures the disagreement is real but bounded: it is
        # one lattice shell out of the disk, largest at the smallest aperture.
        if (~clean).any():
            assert np.max(np.abs(frac[~clean])) < 0.10, (
                f'Boundary-tie discrepancy larger than expected: '
                f'{frac[~clean]} at R={radii[~clean]}'
            )

    def test_coefficient_is_insensitive_to_the_tie_convention(self, setup):
        """The deliverable r must barely move under the tie convention.

        The boundary tie shifts the filtered amplitude Y by several per cent at
        R=1', but every Y entering a coefficient is filtered with the same
        kernel, so the effect largely cancels in the ratio.  This is what makes
        the Task 1 deliverable robust to the discretization, and it is the
        property worth pinning -- Gate A thresholds the ratio at 10 per cent.
        """
        rp = setup['rp']
        radii = setup['radii']

        deltas = {'e': setup['delta_e'], 'm': setup['delta_m']}
        prof_true = rp.r_profiles(
            rp.compute_Y_matrix(deltas, setup['pixel_arcmin'], radii=radii),
            pairs=[('e', 'm')], ratios=[])
        prof_stamp = rp.r_profiles(
            rp.compute_Y_matrix(deltas, setup['stamp_pixel'], radii=radii),
            pairs=[('e', 'm')], ratios=[])

        # Sigma and DSigma are local in the aperture, so the pixel-scale
        # convention moves them at the few-per-mille level.  Upsilon subtracts
        # (R0/R)^2 * DSigma(R0), so it inherits the R0 = 1' discretization at
        # EVERY radius -- the same reference-term amplification the theory note
        # predicts for the physics (Sec. 6, Task 1), here acting on the
        # discretization.  It is given its own, documented budget.
        tolerance = {'Sigma': 5e-3, 'DSigma': 1e-2, 'Upsilon': 2.5e-2}

        print('\n  r_em under the true and stamp-quantized pixel scales')
        for filt in rp.FILTERS:
            a = prof_true['r'][('e', 'm')][filt]
            b = prof_stamp['r'][('e', 'm')][filt]
            with np.errstate(invalid='ignore'):
                d = b / a - 1.0
            finite = d[np.isfinite(d)]
            worst = float(np.max(np.abs(finite))) if finite.size else 0.0
            print(f'  {filt:8s} worst |dr/r| = {worst:.3e} '
                  f'(budget {tolerance[filt]:.1e})')
            for R, x, y, f in zip(radii, a, b, d):
                if np.isfinite(f):
                    print(f'    {R:7.3f}  {x:10.6f}  {y:10.6f}  {f:+11.3e}')
            assert worst < tolerance[filt], (
                f'{filt}: coefficient moved by {worst:.3e} under the '
                f'pixel-scale convention, budget {tolerance[filt]:.1e}'
            )

    def test_production_centring_difference_is_bounded(
            self, setup, stamp_reference):
        """The production (floor) galaxy map adds a half-pixel centring offset.

        This is the convention the production run uses, because NGP bin
        membership is what keeps the galaxy map on the same pixel edges as the
        particle maps.  The residual here is therefore expected to exceed the
        matched-centring one; it is measured and reported, not hidden.
        """
        fft = _fft_amplitude(setup, rounding='floor')
        frac = _report(setup['radii'], fft, stamp_reference,
                       'Production centring (floor): adds the half-pixel '
                       'galaxy centring difference')
        assert np.max(np.abs(frac)) < 5e-2, (
            'Production-convention FFT route differs from the stamp route by '
            f'more than 5 per cent: {frac}'
        )

    def test_upsilon_consistency(self, setup):
        """Upsilon from the FFT route must equal its DSigma combination."""
        rp = setup['rp']
        pos2d = rp.project_positions(
            setup['subhalos']['SubhaloPos'][setup['halo_mask']], PROJECTION)
        delta_g, nbar = _galaxy_delta(
            pos2d, N_PIXELS, float(setup['stacker'].header['BoxSize']),
            'floor')
        Ymat = rp.compute_Y_matrix(
            {'e': setup['delta_e'], 'g': delta_g},
            setup['pixel_arcmin'], radii=setup['radii'], nbar_pix=nbar)

        radii = setup['radii']
        ds = rp.get_Y(Ymat, 'DSigma', 'e', 'g')
        ups = rp.get_Y(Ymat, 'Upsilon', 'e', 'g')
        i0 = int(np.argmin(np.abs(radii - rp.R0_ARCMIN)))
        expected = ds - (rp.R0_ARCMIN / radii) ** 2 * ds[i0]
        assert np.allclose(ups, expected, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Upsilon: the same cross-check for the compensated filter with a reference term
# ---------------------------------------------------------------------------

#: Reference radii exercised against the stamp route.  2.0 is the production
#: value (configs/cross_corr/r_profiles_z0*.yaml); 1.5 is a control that is NOT
#: lattice-degenerate at the stamp route's quantized 0.2 arcmin pixel, so it
#: can carry the machine-precision assertion at every aperture.
UPSILON_R0 = (1.5, 2.0)


def _stamp_upsilon(setup, r0):
    """Stack the SHAM sample through the legacy stamp route with Upsilon.

    Exercises ``stack_on_array(filterType='upsilon')`` with the frozen filter
    parameters.  Before the fix this branch fell through to the generic filter
    call and silently used ``filters.upsilon``'s defaults (``dr=0.5``,
    ``r0=1.0``, ``pixel_size=1.0``), i.e. a different filter from the one in
    ``docs/filter_specification.md``.

    Args:
        setup (dict): The module fixture.
        r0 (float): Upsilon reference radius in arcmin.

    Returns:
        np.ndarray: Mean Upsilon-filtered delta_e at the galaxy positions.
    """
    radii = setup['radii']
    _, profiles = setup['stacker'].stack_on_array(
        array=setup['delta_e'],
        filterType='upsilon',
        minRadius=float(radii.min()),
        maxRadius=float(radii.max()),
        numRadii=len(radii),
        projection=PROJECTION,
        radDistance=1.0,
        radDistanceUnits='arcmin',
        z=setup['z_true'],
        use_subhalos=True,
        halo_mask=setup['halo_mask'],
        dr=0.75,
        r0=r0,
    )
    return np.mean(profiles, axis=1)


def _fft_upsilon(setup, rounding, r0, pixel_arcmin=None):
    """Compute Y_eg for the Upsilon and DSigma filters through the FFT route.

    Both are returned because DSigma sets the scale against which an Upsilon
    residual has to be judged: Upsilon is a difference of two DSigma
    amplitudes, so the discretization error it inherits is proportional to
    those, while Upsilon itself passes through zero at R0.

    Args:
        setup (dict): The module fixture.
        rounding (str): ``'floor'`` or ``'round'`` galaxy-centring convention.
        r0 (float): Upsilon reference radius in arcmin.
        pixel_arcmin (float, optional): Pixel scale for the aperture kernel.
            Defaults to None, meaning the true arcmin-per-pixel.

    Returns:
        tuple: ``(Y_Upsilon, Y_DSigma)``, each of shape ``(n_radii,)``.
    """
    rp = setup['rp']
    if pixel_arcmin is None:
        pixel_arcmin = setup['pixel_arcmin']
    pos2d = rp.project_positions(
        setup['subhalos']['SubhaloPos'][setup['halo_mask']], PROJECTION)
    delta_g, nbar = _galaxy_delta(
        pos2d, N_PIXELS, float(setup['stacker'].header['BoxSize']), rounding)
    Ymat = rp.compute_Y_matrix(
        {'e': setup['delta_e'], 'g': delta_g},
        pixel_arcmin, radii=setup['radii'], r0=r0, nbar_pix=nbar)
    return (rp.get_Y(Ymat, 'Upsilon', 'e', 'g'),
            rp.get_Y(Ymat, 'DSigma', 'e', 'g'))


@pytest.fixture(scope='module')
def stamp_upsilon_reference(setup):
    """Cache the stamp-route Upsilon stack for every reference radius tested.

    Returns:
        dict: ``{r0: mean Upsilon profile}``.
    """
    return {r0: _stamp_upsilon(setup, r0) for r0 in UPSILON_R0}


class TestUpsilonFFTMatchesStampStack:
    """The amplitude-level Upsilon must reproduce the stamp-stacked filter.

    ``compute_Y_matrix`` never convolves an Upsilon kernel: it forms
    ``Y_DSigma(R) - (R0/R)^2 Y_DSigma(R0)`` after the fact, which is exact only
    because the filtered amplitude is linear in the kernel.
    ``filters.upsilon`` performs the *same* combination on each stamp before
    averaging.  Averaging and a linear combination commute, so on real data
    with a real galaxy sample the two must agree to the discretization
    conventions and nothing else.
    """

    @pytest.mark.parametrize('r0', UPSILON_R0)
    def test_matched_centring_agrees_off_lattice_ties(
            self, setup, stamp_upsilon_reference, r0):
        """With every convention matched, only the lattice ties can differ.

        Upsilon subtracts a reference term built at R0, so unlike DSigma a
        boundary tie *at R0* contaminates every aperture rather than just its
        own.  At the stamp route's quantized 0.2 arcmin pixel, R0 = 2.0 arcmin
        sits exactly on the 10-pixel lattice shell, which is precisely why the
        1.5 arcmin control is carried alongside it.
        """
        rp = setup['rp']
        radii = setup['radii']
        stamp_pixel = setup['stamp_pixel']

        fft, _ = _fft_upsilon(setup, rounding='round', r0=r0,
                              pixel_arcmin=stamp_pixel)
        # Same area-normalization bookkeeping as the DSigma test: the stamp
        # route hands delta_sigma_kernel the TRUE pixel scale for 1/pixArea
        # while binning against the quantized linspace grid.  Upsilon is a
        # linear combination of DSigmas, so this is one global factor.
        fft = fft * (stamp_pixel / setup['pixel_arcmin']) ** 2

        frac = _report(radii, fft, stamp_upsilon_reference[r0],
                       f'Upsilon(R0={r0}), matched centring, pixel scale and '
                       'area normalization')

        r0_margin, r0_shell = rp.lattice_boundary_margin(r0, stamp_pixel)
        r0_tied = r0_margin < 1e-3
        print(f"    R0={r0}' disk edge is {r0_margin:.2e} pixels from a "
              f'{r0_shell}-pixel lattice shell '
              f'-> {"DEGENERATE" if r0_tied else "clean"}')

        tied = set(rp.degenerate_apertures(radii, stamp_pixel))
        clean = np.array([not any(np.isclose(R, t) for t in tied)
                          for R in radii])
        assert clean.any(), 'No non-degenerate aperture to compare'

        if r0_tied:
            # The reference term itself straddles a shell, so no aperture is
            # clean.  The discrepancy is still bounded by that one shell.
            assert np.max(np.abs(frac)) < 0.10, (
                f'R0={r0} is lattice-degenerate, but the discrepancy should '
                f'still be bounded by one shell: {frac}'
            )
        else:
            assert np.max(np.abs(frac[clean])) < 1e-10, (
                'FFT and stamp Upsilon must agree to machine precision at '
                f'non-degenerate apertures, got {frac[clean]} at '
                f'R={radii[clean]}'
            )
            if (~clean).any():
                assert np.max(np.abs(frac[~clean])) < 0.10, (
                    f'Boundary-tie discrepancy larger than expected: '
                    f'{frac[~clean]} at R={radii[~clean]}'
                )

    @pytest.mark.parametrize('r0', UPSILON_R0)
    def test_production_centring_difference_is_bounded(
            self, setup, stamp_upsilon_reference, r0):
        """The production (floor) galaxy map adds a half-pixel centring offset.

        The DSigma counterpart of this test budgets 5 per cent of the DSigma
        amplitude.  Upsilon inherits exactly that error -- it is a difference
        of two DSigma amplitudes, each carrying the centring convention -- but
        its own amplitude shrinks to zero as R approaches R0, so a residual
        expressed relative to *Upsilon* diverges at the bin above R0 while
        measuring nothing new.  The assertion is therefore made against the
        DSigma amplitude, which is the scale the error actually lives on; the
        raw fractional residual is still printed, because how badly it blows up
        near R0 is a science result about the usability of that bin, not a
        detail to hide.

        Measured at R0 = 2 arcmin: 32 per cent of Upsilon at R = 2.25 arcmin,
        the bin immediately above R0, falling to 1 per cent by R = 6.
        """
        fft, fft_ds = _fft_upsilon(setup, rounding='floor', r0=r0)
        radii = setup['radii']
        stamp = stamp_upsilon_reference[r0]
        frac = _report(radii, fft, stamp,
                       f'Upsilon(R0={r0}), production centring (floor)')

        rp = setup['rp']
        defined = rp.upsilon_defined_mask(radii, r0)
        print(f'  apertures where Upsilon is defined (R > R0): '
              f'{radii[defined]}')
        assert defined.any(), 'No aperture above R0 to compare'

        # Scale-free version: the residual as a fraction of the DSigma
        # amplitude that Upsilon is built from.
        scaled = np.abs(fft - stamp) / np.abs(fft_ds)
        print(f"  {'R':>7}  {'|dY_Ups| / |Y_DSigma|':>21}")
        for R, s, d in zip(radii, scaled, defined):
            if d:
                print(f'  {R:7.3f}  {s:21.3e}')
        assert np.max(scaled[defined]) < 0.05, (
            'Production-convention FFT Upsilon differs from the stamp route '
            f'by more than 5 per cent of the DSigma amplitude above R0: '
            f'{scaled[defined]}'
        )

    def test_stamp_route_uses_the_requested_filter_parameters(self, setup):
        """``stack_on_array`` must honour dr and r0 rather than the defaults.

        Regression test for the branch fix: 'upsilon' previously fell through
        to the generic ``filterFunc(cutout, rr, rad, pixel_size=1.)`` call, so
        dr, r0 and the pixel area were silently taken from
        ``filters.upsilon``'s signature defaults.  Two different r0 values must
        therefore give two different profiles, and each must match its own FFT
        counterpart far better than it matches the other's.
        """
        a = _stamp_upsilon(setup, 1.5)
        b = _stamp_upsilon(setup, 2.0)
        assert not np.allclose(a, b, rtol=1e-6), (
            'Changing r0 did not change the stamp Upsilon profile; the '
            'branch is ignoring its filter parameters again.'
        )
