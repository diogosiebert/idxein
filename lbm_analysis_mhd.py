#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:34:47 2023

@author: diogo
"""

import sympy as sp
from idxein import *
import numpy as np


rho, tt, cs = sp.symbols("\\rho, \\theta, c_s", real = True)
i = sp.Idx("i")
w = sp.IndexedBase("w")
u_o = IndexedBase("u_o")
u = IndexedBase("u")
xi_o = IndexedBase("\\xi_o")
x = IndexedBase("x")
t = sp.Symbol("t", real = True, positive = True)
c = IndexedBase("c")
TT = sp.Symbol("\\Theta")

idx = lambda n : [ a(i) for i in range(1,n+1) ]

feq = S.Zero
for n in range(3):
    H = HermiteTensor( idx(n), xi_o )
    A = computeMoment( H.subs( { xi_o[k] : sp.sqrt(TT+1)*c[k] + u_o[k] for k in idx(n) } )  , c , a(1) )
    feq +=  A*H / sp.factorial(n)     

dk = KroneckerDelta

Dim = 2

sp.init_printing()

z = sp.Symbol( "z", complex = True)
delta = sp.Symbol( "\\delta", real = True)
ttau = sp.Symbol( "\\tilde\\tau", real = True)
coeff = lambda n : sp.diff( 1/ ( 1 + ttau * ( sp.exp( delta * z) -1 ) / delta), z , n ).subs( {z : 0} ) / sp.factorial(n)

rho, tt, cs = sp.symbols("\\rho, \\theta, c_s", real = True)
T = sp.symbols("\\Theta", real = True)

u = IndexedBase("u")
x = IndexedBase("x")
g = IndexedBase("g")
t = sp.Symbol("t", real = True, positive = True)

a  = lambda n : IdxEin("\\alpha_{}".format(n), range=(1,Dim) )
b  = lambda n : IdxEin("\\beta_{}".format(n), range=(1,Dim) )
e  = lambda n : IdxEin("\\eta_{}".format(n), range=(1,Dim) )

c = IndexedBase("c")
mc = lambda n : rho* computeMoment( sp.expand( sp.Mul( *[ sp.sqrt(T) * cs* c[a(k)] + u[a(k)] for k in range(1,n+1) ] ) ) , c , a(1) )
force =  lambda n : rho * computeMoment( simplifyKronecker( g[b(1)] * sp.diff( sp.Mul( *[ c[ a(n) ] + u[a(n)]  for n in range(1,n+1) ]) , c[b(1)] ) ) , c, a(1) )

m = sp.Matrix( [   mc(i) for i in range(0,5) ] )

NumOfMoments = len(m)
ConservedMoments = 3

def F(n):
    return sp.Matrix( [ force(i) for i in range(0,n) ] )    

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

var = lambda b : [rho, u[b] , T  ] 
J = lambda b, n = None  : ( m if n == None else sp.Matrix( m[:n] ) ).jacobian(  var(b) )

n = len(m)
MLJ = simplifyKronecker( M(n-1) @ L(n) @ J( b(1) , n  ) ).subs( { dk( b(1), a(1)) : 1 } ).xreplace( { b(1) : a(1) } )
#MUJ = simplifyKronecker( M(n-1) @ U(n) @ J( b(1) , n  ) ).subs( { dk( b(1), a(1)) : 1 } ).xreplace( { b(1) : a(1) } )
MUJ = simplifyKronecker( M(n-1) @ U(n) @ J( b(1) , n  ) )
#TG = lambda bt = b(1), et = e(1) : simplifyKronecker( - MLJ.inv() @ MUJ ).xreplace( { a(1) : bt, e(1) : et } )

grad = D( sp.Matrix( [ rho, u[b(1)] , T ] ) , x[ e(1) ]  )
#dt1 = simplifyKronecker( TG(b(1)) @ grad )

#O = simplifyKronecker( ( - L(5) @ J(b(2), 5) @ simplifyKronecker( (MLJ.inv() @ MUJ ) ).xreplace( { a(1) : b(2) } ) + U( 5, e(1) ) @ J( b(1), 5 ) ) @ grad )
#- coeff(1) / ttau *simplifyKronecker( (MLJ.inv() @ MUJ ).xreplace( { a(1) : b(1) } ) @ grad ) - coeff(2) / ttau * simplifyKronecker( MLJ.inv() @ M(3) @ simplifyD( D( simplifyKronecker( U(4, e(2) ) @  O ), x[e(2)] ) )   )

dt0 = simplifyKronecker( MLJ.inv() @ M(n) @ F(n) ) - simplifyKronecker( MLJ.inv() @ MUJ  @ grad )
O = simplifyKronecker( L(n) @ J(b(2), n) @ dt0.xreplace( { a(1) : b(2) } ) ) + simplifyKronecker( ( U(n, e(1) ) @ J( b(2), n) @ grad.xreplace( { b(1) : b(2) } ) ) ) 
