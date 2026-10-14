# Kimberlina dimensional provenance

Depth, contact elevation and horizontal coordinates are in **metres**.

Evidence was checked on 18 September 2026 in Software Underground Subsurface
commit `816db75bad6f1eda71209a4635490d2969eadeb0`:

- [Raw contact table and header](https://github.com/softwareunderground/subsurface/blob/816db75bad6f1eda71209a4635490d2969eadeb0/tests/data/borehole/doggr_jlw_vedder_final.utm.dat): topography and formation fields explicitly say `meters`; projection is UTM zone 11, Clarke 1866/NAD27; projected units are metres.
- [Conversion test](https://github.com/softwareunderground/subsurface/blob/816db75bad6f1eda71209a4635490d2969eadeb0/subsurface/modules/reader/wells/DEP/test_wells/test_welly_to_subsurface.py): `test_aux_operations` loads that contact table and builds `kim_ready.csv`.
- [Conversion helpers](https://github.com/softwareunderground/subsurface/blob/816db75bad6f1eda71209a4635490d2969eadeb0/subsurface/modules/reader/wells/wells_utils.py): depth is collar altitude minus contact elevation; interval tops are shifted prior depths; reversed intervals are repaired with `base=top+0.01`.

All 583 labelled intervals used by GeoBench match numerical reconstruction
within 1e-8 m; all 69 collars agree. Two excluded `topo` records have a known
well-name collision difference, recorded in the verification JSON.

Twenty labelled intervals contain the explicit upstream reversal repair; 17
are nonterminal boundary candidates. The supplementary sensitivity excludes
these 17 boundary targets and donors and refits models with unchanged settings.

This source trace supersedes protocol 2.0 metadata that described dimensional
units as unverified. It changes unit annotation, not the numerical scale of
stored predictions. Original results and their historical code hashes remain
unaltered. Formation interpretation and imported label mapping are separate
from the established dimensional units.
