"""Check scientific separation of feature factors and information boundaries."""
import numpy as np
import pandas as pd
import pytest
from GeoBench.reproducibility.feature_diagnostics import select_block
from GeoBench.reproducibility.support_diagnostics import classify_support


def test_factorial_columns_form_two_disjoint_optional_blocks():
    names=["x","y","z","collar","nearest_other_distance","donors_within_radius","depth","nearest_label:sand","nearest_label:unknown","neighbour_fraction:sand","neighbour_fraction:unknown"]
    blocks={v:set(select_block(names,v,2)) for v in ("raw","geometry","donor","augmented")}
    assert blocks["geometry"]&blocks["donor"]==blocks["raw"]
    assert blocks["geometry"]|blocks["donor"]==blocks["augmented"]==set(range(len(names)))
    assert {names[i] for i in blocks["geometry"]-blocks["raw"]}=={"collar","nearest_other_distance","donors_within_radius","depth"}
    with pytest.raises(ValueError):select_block(names,"made_up_variant",2)


def test_support_distinguishes_missing_and_conflicting_donors():
    data=pd.DataFrame([dict(borehole="A",x=0.,y=0.,top=0.,bottom=10.,label="sand"),dict(borehole="B",x=1.,y=0.,top=0.,bottom=10.,label="clay"),dict(borehole="B",x=1.,y=0.,top=5.,bottom=8.,label="sand")])
    q=pd.DataFrame(dict(query_id=["Q:1","Q:2","Q:3"],borehole=["Q"]*3,x=[2.]*3,y=[0.]*3,depth=[2.,6.,11.],target=["unknown"]*3))
    score=classify_support(data,q,["A","B"])
    assert score.known_donors.tolist()==[2,1,0]
    assert score.nearest_donor_supported.tolist()==[True,False,False]
    changed=q.assign(target=["clay","sand","sand"])
    pd.testing.assert_frame_equal(score,classify_support(data,changed,["A","B"]))
    with pytest.raises(ValueError):classify_support(data,q.assign(borehole="A"),["A","B"])
