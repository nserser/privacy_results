import math
import numpy as np
from scalecert.occupancy import next_batch_probs, enumerate_occupancies, occupancy_probability, expected_collision_pairs

def test_next_probs_sum():
    for r,S,b in [(3,20,5),(0,16,4),(8,8,4)]:
        assert abs(sum(p for _,p in next_batch_probs(r,S,b))-1)<1e-14

def test_enumeration_probability_sum():
    for M,b,k in [(4,1,2),(3,2,2),(4,2,3)]:
        rows=list(enumerate_occupancies(M,b,k))
        assert abs(sum(p for _,p in rows)-1)<1e-13
        for c,p in rows: assert abs(p-occupancy_probability(c,b,k))<1e-15

def test_collision_b1_zero():
    assert expected_collision_pairs(100,1,8)==0
