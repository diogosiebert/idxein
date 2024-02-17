#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 08:44:57 2024

@author: diogo
"""

import numpy as np
import sympy as sp
from idxein import *

sp.init_printing()
delta = sp.Symbol( "\\delta", real = True)
tau = sp.Symbol( "\\tau", real = True)
wc = sp.Symbol("\\omega_c", positive = True ) 
hb = IndexedBase("b")
tst = IndexedBase("\\tilde\\pi")

k = IdxEin("\\kappa", (1,3) )
g = IdxEin("\\gamma", (1,3) )
n = IdxEin("\\nu"   , (1,3) )
b = IdxEin("\\beta" , (1,3))
a = IdxEin("\\alpha", (1,3) )

X = sp.Matrix( [ [ sp.summation( hb[n] * sp.LeviCivita( g, n , a), n ) for g in range(1,4) ] for a in range(1,4) ] )
Q = (delta/(2*tau)+1)*sp.eye(3) + delta/2 * wc * X

qtilde = sp.Matrix( [-6.832541503177296, 8.722743779803830, -22.271144202203111 ] )
q      = sp.Matrix( [-2.868471941359970 , 3.152004879923934 , -8.31366392972883 ] )

val = { delta : 1, wc : 0.2, tau : 0.3 , hb[1] : -0.11, hb[2] : 0.35, hb[3] : 0.27 }

tau = 0.3;
wc =  0.2;
qtilde = np.array(  [-6.832541503177296, 8.722743779803830, -22.271144202203111 ] )
q = np.array([ -1.307776870856611,3.832700944707954, -0.407046111274916 ])
hb = np.array( [-0.11, 0.35, 0.27] )
hb = hb/ np.linalg.norm(hb)
Q = (1/(2*tau) + 1 )*np.eye(3) + 0.5*wc*np.array( [ [    0     ,  hb[2]  , - hb[1]  ],
                       [  -hb[2]  ,    0    , + hb[0]  ],
                       [  +hb[1]  , -hb[0]  ,   0      ] ] )