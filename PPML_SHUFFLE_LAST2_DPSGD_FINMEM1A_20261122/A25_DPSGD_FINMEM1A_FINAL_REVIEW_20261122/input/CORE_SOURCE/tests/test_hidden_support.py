import numpy as np
from scalecert.hidden_support import evolve_hidden_epoch, evolve_fixed_counts
from scalecert.occupancy import enumerate_occupancies

class Toy:
    def __init__(self):
        self.K={0:np.array([[.9,.2],[.1,.8]]),1:np.array([[.7,.1],[.3,.9]]),2:np.array([[.5,.05],[.5,.95]])}
    def transition(self,v,c): return self.K[int(c)]@v,0.0

def test_hidden_dp_matches_explicit_enumeration():
    M,b,k=3,2,2; q=np.array([.4,.6]); toy=Toy()
    r=evolve_hidden_epoch(q,toy.transition,M,b,k)
    explicit=np.zeros(2)
    for counts,p in enumerate_occupancies(M,b,k):
        rr=evolve_fixed_counts(q,toy.transition,counts)
        explicit += p*rr.mass
    assert np.max(np.abs(r.mass-explicit))<1e-14
    assert abs(r.mass.sum()-1)<1e-14
