"""Scientific checks for the two new, bounded evaluation extensions."""
import numpy as np
from GeoBench.reproducibility.ordered_projection import project_column


def test_projection_known_pooled_block_and_nonexpansion():
    p=np.array([100.,80.,90.,50.]);r=np.arange(4)
    projected=project_column(p,r)
    np.testing.assert_allclose(projected,[100.,85.,85.,50.])
    truth=np.array([105.,93.,72.,40.])
    assert np.sum((projected-truth)**2)<=np.sum((p-truth)**2)
    np.testing.assert_allclose(project_column(projected,r),projected)


def test_projection_respects_source_identity_not_input_row_order():
    p=np.array([100.,80.,90.,50.]);r=np.arange(4);perm=np.array([3,0,2,1])
    np.testing.assert_allclose(project_column(p[perm],r[perm]),project_column(p,r)[perm])


def test_unknown_order_is_rejected_instead_of_derived_from_truth():
    import pytest
    with pytest.raises(ValueError):project_column([1.,2.],[np.nan,np.nan])
    with pytest.raises(ValueError):project_column([1.,2.],[1,1])
