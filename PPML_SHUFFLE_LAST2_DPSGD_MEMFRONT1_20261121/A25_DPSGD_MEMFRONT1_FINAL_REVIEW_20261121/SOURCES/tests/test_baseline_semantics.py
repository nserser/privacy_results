from scalecert.baseline_fs import parse_epsilon_range


def test_epsilon_range_preserves_lower_upper_semantics():
    x=parse_epsilon_range((1.25,0.75))
    assert x['epsilon_lower']==0.75
    assert x['epsilon_upper']==1.25
    y=parse_epsilon_range({'epsilon_lower':0.4,'epsilon_upper':0.9})
    assert y['epsilon_lower']==0.4
    assert y['epsilon_upper']==0.9
