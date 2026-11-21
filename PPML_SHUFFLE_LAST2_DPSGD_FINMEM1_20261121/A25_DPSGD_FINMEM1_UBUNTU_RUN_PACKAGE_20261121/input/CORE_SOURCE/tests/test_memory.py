import numpy as np
from scalecert.memory import adaptive_impulse_curve

class Decay:
    def transition(self,v,c):
        # tiny 2-state toy; c=1 creates a perturbation, c=0 contracts it
        K=np.array([[.9,.1],[.1,.9]]) if c==0 else np.array([[.8,.1],[.2,.9]])
        return K@v,0.0

def chi(a,b):
    return float(np.sum((a-b)**2/np.maximum(b,1e-30)))

def test_adaptive_memory_tail_terminates():
    d,diag=adaptive_impulse_curve(np.array([.5,.5]),Decay().transition,4,chi,tail_ratio_tol=1e-4,max_age=128)
    assert diag['tail_complete']
    assert len(d)>=4
