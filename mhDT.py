#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:16:45 2024

@author: diogo
"""

import sympy as sp
from idxein import *
import numpy as np

dk = KroneckerDelta

def MaxwellStress( i , j ):
    return B[i]*B[j] - B[ g(1) ]*B[ g(1 )] /2  * KroneckerDelta( i , j)

def computeMomentMagnetic(exp, variable, index):
    
    term = exp.expand()    
    if ( term.func == sp.Add ):
        return  simplifyKronecker( sp.Add( *(  computeMomentMagnetic(arg, variable, index) for arg in term.args) ) )
    
    term = unroll(exp)
    if (term.func == sp.Mul ):
        replace = [ arg for arg in term.args if isinstance(arg, Indexed) if (arg.base == variable) if isinstance(arg.indices[0],IdxEin) if arg.indices[0].compatible(index) ]
        keep    = [ arg for arg in term.args if not( arg in replace ) ]
        if len(replace) == 2:
            keep.append( - MaxwellStress( *[ arg.indices[0]  for arg in replace ] ) )
        else:
            return S.Zero
        return unroll( sp.Mul( *keep ) )
    elif( term.is_Number ): return S.Zero
    return computeMomentMagnetic( sp.Mul( term , S.One , evaluate=False) , variable, index)

sp.init_printing()
Dim = 3

a  = lambda n : IdxEin("\\alpha_{}".format(n), range=(1,Dim) )
b  = lambda n : IdxEin("\\beta_{}".format(n), range=(1,Dim) )
e  = lambda n : IdxEin("\\eta_{}".format(n), range=(1,Dim) )
g  = lambda n : IdxEin("\\gamma_{}".format(n), range=(1,Dim) )

rho, tt, cs = sp.symbols("\\rho, \\theta, c_s", real = True)
i = sp.Idx("i")
B = IndexedBase("B")
c = IndexedBase("c")
u = IndexedBase("u")
xi = IndexedBase("\\xi")
x = IndexedBase("x")
t = sp.Symbol("t", real = True, positive = True)
TT = sp.Symbol("\\Theta")

ttau = sp.Symbol(r"\tilde \tau")
delta = sp.Symbol(r"\delta")
cc = sp.Symbol("cc")
uu = sp.Symbol("uu")
cu = sp.Symbol("cu")

afactor = sp.Symbol("a")
a2tt1 = afactor**2 * TT - 1

ifeq = unroll( sp.expand(  sp.nsimplify( rho * ( 1. +  cu + (1./2.)*(  cu * cu  -  uu + a2tt1 * ( cc - 3)) + (1./6.) *  cu * (  cu * cu - 3. *  uu + 3. * a2tt1  * ( cc - 5. ))  + (15./8.) * a2tt1  * a2tt1  + (5./4.)*a2tt1  *  uu + (1./8.) *  uu * uu - (7./4.)*a2tt1  *  cu * cu - (1./4.) *  uu * cu * cu - (5./4.) * a2tt1 *a2tt1 *  cc - (1./4.) * a2tt1  *  uu * cc + (1./4.)*a2tt1  *  cu * cu * cc + (1./28.) *  uu * cu * cu * cc + (1./8.)*a2tt1 *a2tt1 *  cc * cc - (1./280.) *  uu * uu * cc * cc ) ).subs( {  cc**2 :  xi[b(1)]*xi[b(1)]*xi[b(2)]*xi[b(2)]  }  ).subs( {  cc :  xi[b(3)]*xi[b(3)]  }  ).subs( {  cu**2 :  afactor**2 * xi[b(4)]*u[b(4)]*xi[b(5)]*u[b(5)]  }  ).subs( {  cu : afactor *xi[b(6)]*u[b(6)]  }  ).subs( {  uu**2 :  afactor**4 * u[b(7)]*u[b(7)]*u[b(8)]*u[b(8)] }  ).subs( {  uu : afactor** 2 * u[b(9)]*u[b(9)] }  ) ) ) 
mc = lambda n : unroll( computeMoment( ifeq * (S.One if n == 0 else np.prod( [ xi[a(k)] / afactor for k in range(1,n+1) ] ) ), xi ) )

idx = lambda n : [ a(i) for i in range(1,n+1) ]
m = sp.Matrix( [   mc(i) for i in range(0,5) ] )

def F(n):
    return sp.Matrix( [ 0 for i in range(0,n) ] )    

def M(n, conserved = 3):
    M = np.zeros( [ conserved ,  n ] , dtype = object)
    np.fill_diagonal( M , [1,1, dk( a(1), a(2) ) ] )
    return sp.Matrix( M )

def L(n):
    L = np.zeros( [ n-1, n ] , dtype = object)
    np.fill_diagonal( L, 1)
    return sp.Matrix(L)

def U(n, et = e(1) ):
    U = np.zeros( [ n-1, n] , dtype = object)
    np.fill_diagonal( U[:,1:], [ dk(a(k),et ) for k in range(1,n) ] )
    return sp.Matrix(U)

var = lambda b : [rho, u[b] , TT ] 
J = lambda b, n = None  : ( m if n == None else sp.Matrix( m[:n] ) ).jacobian(  var(b) )


MLJ = simplifyKronecker( M( len(m) - 1 ) @ L( len(m) )  @ m.jacobian( var( b(1) ) ) ).subs( { dk( a(1), b(1)) : 1 } ).xreplace( { b(1) : a(1) } )
MUJ = simplifyKronecker( M( len(m) -1  ) @ U( len(m) ) @ J( b(1) ) )
grad = D( sp.Matrix( [ rho, u[b(1)] , TT ] ) , x[ e(1) ]  )
dt0 = simplifyKronecker( - MLJ.inv() @ MUJ  @ grad ).xreplace( { a(1) : b(1) } )

MDO = simplifyKronecker( M(len(m) - 2) @ U( len(m) - 1 , e(2) ) @ ( L( len(m) ) @ J( b(1) ) @ dt0 + U( len(m) , e(1) ) @ J( b(1) ) @ grad ) )




# PI_ab0 = sp.Matrix( [ mc(2)] ).jacobian( [rho, u[b(2)] , B[b(2)] ] ) @ ( dt0B.xreplace( { a(1) : b(2) } ) )
# PI_ab0 = simplifyKronecker(  sp.expand( PI_ab0[0] ) )
# PI_ab1 = simplifyKronecker(  sp.Matrix( [ mc(3)] ).jacobian( [rho, u[b(2)] , B[b(2)] ] ) * dk( a(3), e(1) ) ) @ ( grad.xreplace( { b(1) : b(2)  } ) )
# PI_ab1 = simplifyKronecker(  sp.expand( PI_ab1[0] ) )
# PI_ab = PI_ab1 + PI_ab0 

# PI_ab.xreplace( {b(1) :  IdxEin("\\eta")  } ).xreplace( {e(1) :  IdxEin("\\eta")  } )
