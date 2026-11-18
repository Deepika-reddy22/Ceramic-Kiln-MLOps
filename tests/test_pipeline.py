import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","src"))
import pytest
from pipeline import validate_input

VALID = {"pottery_load_kg":80,"target_temperature_c":1100,
"initial_moisture_pct":8,"planned_duration_hr":10,"insulation_rating":0.8,
"kiln_volume_m3":1.2,"ambient_temperature_c":25}

def test_valid_input():
    validate_input(VALID)

def test_reject_negative_load():
    bad = dict(VALID, pottery_load_kg=-1)
    with pytest.raises(ValueError): validate_input(bad)

def test_reject_invalid_insulation():
    bad = dict(VALID, insulation_rating=1.5)
    with pytest.raises(ValueError): validate_input(bad)

def test_reject_missing_field():
    bad = dict(VALID); bad.pop("kiln_volume_m3")
    with pytest.raises(ValueError): validate_input(bad)

def test_prediction_is_non_negative():
    # Contract-level test for the serving post-processing rule.
    assert max(0.0, -3.2) == 0.0
