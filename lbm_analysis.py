#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:34:47 2023

@author: diogo
"""

import sympy as sp
from idxein import *
import numpy as np

dk = KroneckerDelta

Dim = 2

sp.init_printing()

z = sp.Symbol( "z", complex = True)
delta = sp.Symbol( "\\delta", real = True)
ttau = sp.Symbol( "\\tilde\\tau", real = True)
coeff = lambda n : sp.diff( 1/ ( 1 + ttau * ( sp.exp( delta * z) -1 ) / delta), z , n ).subs( {z : 0} ) / sp.factorial(n)

rho, tt, cs = sp.symbols("\\rho, \\theta, c_s", real = True)
u = IndexedBase("u")
x = IndexedBase("x")
t = sp.Symbol("t", real = True, positive = True)

a  = lambda n : IdxEin("\\alpha_{}".format(n), range=(1,Dim) )
b  = lambda n : IdxEin("\\beta_{}".format(n), range=(1,Dim) )
e  = lambda n : IdxEin("\\eta_{}".format(n), range=(1,Dim) )

c = IndexedBase("c")
mc = lambda n : rho* computeMoment( sp.expand( sp.Mul( *[ sp.sqrt(tt+1) * cs* c[a(k)] + u[a(k)] for k in range(1,n+1) ] ) ) , c , a(1) )

m = sp.Matrix( [ mc(i) for i in range(0,5) ] )

NumOfMoments = len(m)
ConservedMoments = 3

def M(n):
    M = np.zeros( [ 3,  n ] , dtype = object)
    np.fill_diagonal( M , [1,1,dk(a(1),a(2))] )
    return sp.Matrix( M )

def L(n):
    L = np.zeros( [ n-1, n ] , dtype = object)
    np.fill_diagonal( L, 1)
    return sp.Matrix(L)

def U(n, et = e(1) ):
    U = np.zeros( [ n-1, n] , dtype = object)
    np.fill_diagonal( U[:,1:], [ dk(a(k),et ) for k in range(1,n) ] )
    return sp.Matrix(U)

var = lambda b : [rho, u[b] , tt  ] 
J = lambda b, n = None  : ( m if n == None else sp.Matrix( m[:n] ) ).jacobian(  [rho, u[b] , tt  ] )

n = len(m)
MLJ = simplifyKronecker( M(n-1) @ L(n) @ J( b(1) , n  ) ).subs( { dk( b(1), a(1)) : 1 } ).xreplace( { b(1) : a(1) } )
#MUJ = simplifyKronecker( M(n-1) @ U(n) @ J( b(1) , n  ) ).subs( { dk( b(1), a(1)) : 1 } ).xreplace( { b(1) : a(1) } )
MUJ = simplifyKronecker( M(n-1) @ U(n) @ J( b(1) , n  ) )
TG = lambda bt = b(1), et = e(1) : simplifyKronecker( - MLJ.inv() @ MUJ ).xreplace( { a(1) : bt, e(1) : et } )

grad = lambda e,b : sp.Matrix( [ D(v, x[e] ) for v in var(b) ] )
dt   = lambda b : sp.Matrix( [ D(v,t) for v in var( b ) ] )

sp.Eq( dt(b(1)), simplifyKronecker( TG(b(1)) @ grad( e(1), b(1) ) ) )
grad = D( sp.Matrix( [ rho, u[b(1)] , tt ] ) , x[ e(1) ]  )
O = U( 4, e(2) ) @ simplifyKronecker( ( - L(5) @ J(b(2), 5) @ simplifyKronecker( (MLJ.inv() @ MUJ ) ).xreplace( { a(1) : b(2) } ) + U( 5, e(1) ) @ J( b(1), 5 ) ) @ grad )


sp.expand( simplifyD( simplifyKronecker( M(3) @ simplifyKronecker( simplifyD( D( simplifyKronecker( O ) , x[e(2)] ) ) ) )[2]) )