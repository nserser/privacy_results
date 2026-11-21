import math
from scalecert.gaussian import transcript_chi2_exact, transcript_chi2_b1, geometric_weights, final_chi2_exact_moments, final_chi2_k1_closed

def test_transcript_b1():
    for M in [8,16,32]:
        for k in [1,2,3]:
            assert abs(transcript_chi2_exact(M,1,k,.3)-transcript_chi2_b1(M,k,.3))<1e-10

def test_final_moment_k1_matches_closed():
    for M in [8,16,32]:
        for rho in [0,.5,.9,1.0]:
            w=geometric_weights(M,rho)
            a,_=final_chi2_exact_moments(M,1,1,w,.25)
            b=final_chi2_k1_closed(M,w,.25)
            assert abs(a-b)<1e-11

def test_no_forgetting_k1_formula():
    M=32; mu=.2; w=geometric_weights(M,1.0)
    a,_=final_chi2_exact_moments(M,1,1,w,mu)
    assert abs(a-math.expm1(mu*mu/M))<1e-12
