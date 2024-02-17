#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:13:33 2023

@author: diogo
"""


import sympy as sp
from idxein import *

sp.init_printing()

rho, tt, cs = sp.symbols("\\rho, \\theta, c_s", real = True)
T = sp.symbols("\\Theta", real = True)
eps = sp.LeviCivita 

u = IndexedBase("u")
xi = IndexedBase("\\xi")
x = IndexedBase("x")
t = sp.Symbol("t", real = True, positive = True)
c = IndexedBase("c")
g = IndexedBase("g")
B = IndexedBase("B")
af = sp.Symbol("a")
A = IndexedBase("a", symmetric = True)
i = sp.Idx("i")

wo = sp.Symbol("\\omega_o")
tau = IndexedBase("tau", symmetric = True)
q = IndexedBase("q")
hb = IndexedBase("b")

a  = lambda n : IdxEin("\\alpha_{}".format(n), range=(1,3) )
b  = lambda n : IdxEin("\\beta_{}".format(n), range=(1,3) )
e  = lambda n : IdxEin("\\eta_{}".format(n), range=(1,3) )


Fq = - hb[e(2) ]* eps( e(1), e(2), e(3) ) * tau[ e(1), e(4) ] * xi[ e(3) ] * xi[ e(4) ]  - hb[ e(1) ] * ( 2 * tau[e(2), a(1) ] * u[e(3)]  * eps(e(2) ,e(1), e(3)) +    2* q[e(2)] * eps(e(2) , e(1) , a(1))+  2*tau[e(2),e(3)] * u[e(3)] * eps(e(2) ,e(1), a(1) ) ) * (xi[e(4)] * xi[e(4)] * xi[a(1)]- 5 *  xi[a(1)]) / 10

sp.pretty_print( computeMoment( Fq * ( xi[a(2)] - u[a(2)]) * ( xi[a(2)] - u[a(2)]) * (xi[a(3)] - u[a(3)]) , xi, a(1) ) )