from __future__ import annotations
import numpy as np
import cupy as cp

_SRC=r'''extern "C" __global__ void gather_kahan(
 const double* a,const double* b,const long long* offsets,const double* weights,
 const long long* diagonal,const long long* diag_order,long long n,long long outn,double* out) {
 long long o=(long long)blockDim.x*blockIdx.x+threadIdx.x; if(o>=outn) return;
 double total=0.0,comp=0.0;
 // Mirror NumPy implementation's diagonal Kahan updates: shifts descending, stable i order.
 if(weights[0]==0.0){
   for(long long t=0;t<n;t++){long long i=diag_order[t];if(i+diagonal[i]==o){double v=a[i]*b[i];double y=v-comp;double q=total+y;comp=(q-total)-y;total=q;}}
 } else {
   long long shift=offsets[0];double w=weights[0];long long count=n;
   long long i=o-shift;
   if(i>=0&&i<count){double mass=a[i]*b[i];double up=w*mass;double prev=0.0;if(i>0)prev=w*(a[i-1]*b[i-1]);double v=mass-up+prev;double y=v-comp;double q=total+y;comp=(q-total)-y;total=q;}
   else if(i==count){double mass=a[count-1]*b[count-1];double v=w*mass;double y=v-comp;double q=total+y;comp=(q-total)-y;total=q;}
 }
 for(long long d=1;d<n;d++){
   long long c=n-d;long long s0=offsets[d],s1=offsets[n+d];double w0=weights[d],w1=weights[n+d];
   if(w0==0.0&&w1==0.0){
     long long i0=o-s0,i1=o-s1;bool v0=(i0>=0&&i0<c),v1=(i1>=0&&i1<c);
     if(v0&&v1&&i0<=i1){double v=a[i0]*b[i0+d];double y=v-comp;double q=total+y;comp=(q-total)-y;total=q;double u=a[i1+d]*b[i1];y=u-comp;q=total+y;comp=(q-total)-y;total=q;}
     else if(v0&&v1){double v=a[i1+d]*b[i1];double y=v-comp;double q=total+y;comp=(q-total)-y;total=q;double u=a[i0]*b[i0+d];y=u-comp;q=total+y;comp=(q-total)-y;total=q;}
     else if(v0){double v=a[i0]*b[i0+d];double y=v-comp;double q=total+y;comp=(q-total)-y;total=q;}
     else if(v1){double v=a[i1+d]*b[i1];double y=v-comp;double q=total+y;comp=(q-total)-y;total=q;}
   } else {
     long long i=o-s0;
     if(i>=0&&i<c){double mass=a[i]*b[i+d];double up=w0*mass;double prev=0.0;if(i>0)prev=w0*(a[i-1]*b[i-1+d]);double v=mass-up+prev;double y=v-comp;double q=total+y;comp=(q-total)-y;total=q;}
     else if(i==c){double mass=a[c-1]*b[c-1+d];double v=w0*mass;double y=v-comp;double q=total+y;comp=(q-total)-y;total=q;}
     i=o-s1;
     if(i>=0&&i<c){double mass=a[i+d]*b[i];double up=w1*mass;double prev=0.0;if(i>0)prev=w1*(a[i-1+d]*b[i-1]);double v=mass-up+prev;double y=v-comp;double q=total+y;comp=(q-total)-y;total=q;}
     else if(i==c){double mass=a[c-1+d]*b[c-1];double v=w1*mass;double y=v-comp;double q=total+y;comp=(q-total)-y;total=q;}
   }
 }
 out[o]=total;
}
'''
_kernel=cp.RawKernel(_SRC,'gather_kahan',options=('--std=c++11','--fmad=false'))
def deterministic_geometric_kernel(*,pmf_base,pmf_scaled,bin_offsets,weights,diagonal_bins,output_size):
 a=np.ascontiguousarray(pmf_base,dtype=np.float64);b=np.ascontiguousarray(pmf_scaled,dtype=np.float64)
 bins=np.ascontiguousarray(bin_offsets,dtype=np.int64);w=np.ascontiguousarray(weights,dtype=np.float64);diag=np.ascontiguousarray(diagonal_bins,dtype=np.int64)
 n=a.size;diag_order=np.argsort(-diag,kind='stable').astype(np.int64,copy=False)
 da,db,do,dw,dd,dord=[cp.asarray(q) for q in (a,b,bins,w,diag,diag_order)];out=cp.empty(output_size,dtype=cp.float64)
 threads=128;blocks=(output_size+threads-1)//threads
 _kernel((blocks,),(threads,),(da,db,do,dw,dd,dord,np.int64(n),np.int64(output_size),out))
 return cp.asnumpy(out)
