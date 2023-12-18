#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:49:35 2023

@author: diogo
"""


import sympy as sp
from idxein import *

sp.init_printing()

rho, tt, cs = sp.symbols("\\rho, \\theta, c_s", real = True)
T = sp.symbols("\\Theta", real = True)

u = IndexedBase("u")
x = IndexedBase("x")
t = sp.Symbol("t", real = True, positive = True)
c = IndexedBase("c")
g = IndexedBase("g")
af = sp.Symbol("a")
A = IndexedBase("a")
i = sp.Idx("i")

a  = lambda n : IdxEin("\\alpha_{}".format(n), range=(1,Dim) )
b  = lambda n : IdxEin("\\beta_{}".format(n), range=(1,Dim) )
e  = lambda n : IdxEin("\\eta_{}".format(n), range=(1,Dim) )

def computeMomentNeq(exp, variable, index):
    
    if (isinstance( exp, sp.core.Number )):
        return S.Zero
    term = exp.expand()
    if ( term.func == sp.Add ):
        return  simplifyKronecker( sp.Add( *(  computeMomentNeq(arg, variable, index) for arg in term.args) ) )
    
    if (term.func == sp.Mul ):
        replace = [ arg for arg in term.args if isinstance(arg, Indexed) if (arg.base == variable) if isinstance(arg.indices[0],IdxEin) if arg.indices[0].compatible(index) ]
        keep    = [ arg for arg in term.args if not( arg in replace ) ]
        idx = [ arg.indices[0]  for arg in replace ]
        if ( len(idx) > 1):
            keep.append( A[idx] )
        else:
            keep.append( S.Zero )
        return unroll( sp.Mul( *keep ) )
    else:
        return computeMoment( sp.Mul( term , S.One , evaluate=False) , variable, index)

Dim = 3

N = 3
idx = tuple( a(n) for n in range(1,N+1)  )

H = lambda idx : HermiteTensor( idx  ,  x ).subs(  { x[ k ] : af * c[ k ] for k in idx } ) 
a_eq = lambda idx : rho * computeMoment( HermiteTensor(  idx  , x ).subs(  { x[ k ] :  (sp.sqrt( tt) * c[ k ] + af * u[ k ] )  for k in idx } ) , c, a(1) if len(idx) == 0 else idx[0]  )
a_neq = lambda idx : computeMomentNeq( HermiteTensor(  idx  , x ).subs(  { x[ k ] :  (sp.sqrt( tt) * c[ k ] + af * u[ k ] )  for k in idx } ) , c, a(1) if len(idx) == 0 else idx[0]  )

feq = S.Zero
N = 2
for n in range(N+1):
    idx = tuple( map( a, np.arange(1,n+1) ) ) 
    feq += a_eq(idx) * H( idx ) / sp.factorial(n)

Fneq = S.Zero
N = 4
for n in range(1,N+1):
    idx = tuple( map( a, np.arange(1,n+1) ) ) 
    Fneq += af * g[idx[0] ] * a_neq(idx[1:]) * H( idx ) / sp.factorial(n)
Fneq = simplifyByPermutation( replaceIndeces(  simplifyDeviatoric( simplifyKronecker(Fneq ), A) ), A)

Feq = S.Zero
N = 2
for n in range(1,N+1):
    idx = tuple( map( a, np.arange(1,n+1) ) ) 
    Feq += af * g[idx[0] ] * a_eq(idx[1:]) * H( idx ) / sp.factorial(n)
Feq = simplifyByPermutation( replaceIndeces(  simplifyDeviatoric( simplifyKronecker(Feq ), A) ), A)

Feq = S.Zero
N = 2
for n in range(1,N+1):
    idx = tuple( map( a, np.arange(1,n+1) ) ) 
    Feq += af * g[idx[0] ] * a_eq(idx[1:]) * H( idx ) / sp.factorial(n)
Feq = simplifyByPermutation( replaceIndeces(  simplifyDeviatoric( simplifyKronecker(Feq ), A) ), A)


