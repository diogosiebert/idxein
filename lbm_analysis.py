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

rho, tt, cs = sp.symbols("\\rho, \\theta, c_s", real = True)
u = IndexedBase("u")
x = IndexedBase("x")

a  = lambda n : IdxEin("\\alpha_{}".format(n), range=(1,Dim) )
b  = lambda n : IdxEin("\\beta_{}".format(n), range=(1,Dim) )
e  = lambda n : IdxEin("\\eta_{}".format(n), range=(1,Dim) )

m = sp.Matrix( [ rho, 
                 rho * u[a(1) ] ,
                 rho * u[a(1) ] * u[a(2) ] + rho * cs**2 *(tt+1) *  dk( a(1), a(2) )  ,
                 rho * u[a(1) ] * u[a(2) ]  * u[a(3) ] + rho * cs**2 *(tt+1) * ( u[a(3) ] * dk( a(1), a(2) ) +  u[a(1) ] * dk( a(2), a(3) ) +   u[a(2) ] * dk( a(1), a(3) ) ) ] )

NumOfMoments = len(m)
ConservedMoments = 3

M = np.zeros( [ ConservedMoments, NumOfMoments-1] , dtype = object)
np.fill_diagonal( M, [1,1,dk(a(1),a(2))] )
M = sp.Matrix( M )

L = np.zeros( [ NumOfMoments-1, NumOfMoments] , dtype = object)
np.fill_diagonal( L, 1)
L = sp.Matrix(L)

U = np.zeros( [ NumOfMoments-1, NumOfMoments] , dtype = object)
np.fill_diagonal( U[:,1:], [ dk(a(n),e(1)) for n in range(1,NumOfMoments) ] )
U = sp.Matrix(U)

var = lambda b : [rho, u[b] , tt  ] 
Jb = lambda b : m.jacobian(  [rho, u[b] , tt  ] )

# Simplifica Delta de Kronecker e fazer a substituição do símbolo a_1 por beta
MLJ = simplifyKronecker( M @ L @ Jb( b(1) ) ).subs( { dk( b(1), a(1)) : 1 } ).xreplace( { b(1) : a(1) } )
MUJ = simplifyKronecker( M @ U @ Jb( b(1) ) ).subs( { dk( b(1), a(1)) : 1 } ).xreplace( { b(1) : a(1) } )

grad = lambda e,b : sp.Matrix( [ D(v, x[e] ) for v in var(b) ] )

TG = lambda b : simplifyKronecker( - MLJ.inv() @ MUJ ).xreplace( { a(1) : b } )



