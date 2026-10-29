"""make_pk_fstar_obs_column_dsigma_lensing.py
==========================================
make_pk_fstar_obs_column.py with the lensing panel through the v2 filter
(option C, ``DSigma_lensing``): the single-column S(k) and lensing
f_gas(theta) / (Omega_b/Omega_m) figure with the lensing v2.1 data, as drawn
on 2026-09-30 with the v1 filter (``DSigma``) and the lensing data of
``spec_Y3/``. Made for a research proposal (2026-10-29).

Two changes, made from here (``apply_patches``) so that the existing scripts
stay as they are:

* ``compute_stellar_maps.resolve_stack_settings`` accepts the v2 filter on
  both profiles. Like ``DSigma`` it is a weighted pixel sum, linear in the
  map, so the f(s) algebra of the stellar ends holds; the explicit-map checks
  of the 2026-10-26 stacking runs agree with it to ~1e-14.
* The IllustrisTNG colours are those of the 2026-09-30 figure, before
  88b6d47 added TNG100-1 to make_pk_stellar.py's colour list.

It reads the stacks tagged ``lens_dsigma_lensing`` and otherwise runs
make_pk_fstar_obs_column.py unchanged (same arguments and output naming).

Run from the scripts/ directory (light; login node is fine):
    python unbound_gas/make_pk_fstar_obs_column_dsigma_lensing.py \\
        -p configs/unbound_gas/pk_fstar_obs_z05_n5e-4_dsigma_lensing_v2.1.yaml
"""

from typing import Optional

import compute_stellar_maps as csm
import make_pk_fstar_obs_column as column
import make_pk_stellar as mps

V2_FILTER = 'DSigma_lensing'
# make_pk_stellar.py's IllustrisTNG colour list before 88b6d47 (2026-10-07).
TNG_COLOURS_2026_09_30 = ('twilight', ['TNG300-1', 'Illustris-1'])

_resolve_dsigma_only = csm.resolve_stack_settings


def resolve_stack_settings_v2(nb_stack: dict, overrides: Optional[dict] = None) -> dict:
    """``resolve_stack_settings`` for the v2 filter on both profiles.

    Runs every other check of the original on the block with ``DSigma`` in
    place of the v2 filter, then puts the v2 filter back, so the result (stored
    with the stacks and compared with them) names the filter used.

    Raises:
        ValueError: If the two profiles do not both use the v2 filter.
    """
    merged = {**nb_stack, **(overrides or {})}
    filters = (merged.get('filter_type'), merged.get('filter_type_2'))
    if filters != (V2_FILTER, V2_FILTER):
        raise ValueError(f"filter_type, filter_type_2 = {filters}: this script needs "
                         f"{V2_FILTER!r} for both profiles (else use make_pk_fstar_obs_column.py)")
    as_dsigma = dict(filter_type='DSigma', filter_type_2='DSigma')
    s = _resolve_dsigma_only({**nb_stack, **as_dsigma},
                             {**overrides, **as_dsigma} if overrides else None)
    s.update(filter_type=V2_FILTER, filter_type_2=V2_FILTER)
    return s


def apply_patches() -> None:
    """Accept the v2 filter in compute_stellar_maps and restore the 2026-09-30 colours.

    ``lensing_settings`` looks up ``resolve_stack_settings`` in its module at
    call time, so the patch reaches every script that imported it by name.
    """
    csm.resolve_stack_settings = resolve_stack_settings_v2
    mps._SUITE_COLOURS['IllustrisTNG'] = TNG_COLOURS_2026_09_30


if __name__ == '__main__':
    apply_patches()
    column.main()
