#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:49:37 2023

@author: diogo
"""

import sympy as sp
from idxein import *

sp.init_printing()
delta = sp.Symbol( "\\delta", real = True)
ttau = sp.Symbol( "\\tilde\\tau", real = True)
wc = sp.Symbol("\\omega_c", positive = True ) 
hb = IndexedBase("b")
tst = IndexedBase("\\tilde\\pi")
tau = IndexedBase("\\pi")
k = IdxEin("\\kappa", (1,3) )
g = IdxEin("\\gamma", (1,3) )
n = IdxEin("\\nu"   , (1,3) )
b = IdxEin("\\beta" , (1,3))
a = IdxEin("\\alpha", (1,3) )
subsym = { tau[2,1] : tau[1,2] , tau[3,1] : tau[1,3], tau[3,2] : tau[2,3] , tau[3,3] : - tau[1,1] - tau[2,2] }
# Eq = lambda x, y :  sp.Eq( ((1+delta/(2*tau)) * tau[ a, b ] - sp.summation( wc * delta / 2 * ( sp.LeviCivita( g,  n, a) *hb[n] *  tau[ b, g ] + sp.LeviCivita( g,  n, b) *hb[n] *  tau[ a, g ] ) , g , n )).xreplace( {a : x, b: y} ) , tst[x,y] )
# EqSet = [ Eq(1,1), Eq(1,2), Eq(1,3).subs(subsym), Eq(2,2), Eq(2,3), Eq(3,3).subs( subsym ) ]

LHS = lambda x,y : ((1+delta/(2*ttau)) * tau[ a, b ]+  sp.summation( wc * delta / 2 * ( sp.LeviCivita( g,  n, a) *hb[n] *  tau[ b, g ] + sp.LeviCivita( g,  n, b) *hb[n] *  tau[ a, g ] ) , g , n )).xreplace( {a : x, b: y} )

A = sp.Matrix( [ LHS(1,1).subs( subsym  ), LHS(1,2).subs( subsym  ), LHS(2,2).subs( subsym  ) , LHS(2,3).subs( subsym  ),  LHS(1,3).subs( subsym  )  ] )

Tsys = A.jacobian( ( tau[1,1], tau[1,2] , tau[2,2], tau[2,3],  tau[1,3]  ) )


Tmod = Tsys.subs( { delta * wc * hb[1]: hb[1]  ,  delta * wc * hb[2] : hb[2] ,  delta * wc * hb[3] : hb[3],   delta / (2* ttau ) + 1 : sp.Symbol( "\\lambda" )  } )
TmodInv = Tmod.inv()

Tinv = TmodInv.subs( { hb[1] : delta * wc * hb[1]  ,  hb[2]: delta * wc * hb[2] ,  hb[3] : delta * wc * hb[3] ,   sp.Symbol( "\\lambda" ) : delta / (2* ttau ) + 1 } )
