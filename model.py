#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 20:47:01 2023

@author: diogo
"""
import itertools 
import sympy as sp
import numpy as np
C = np.array( [ [0,0,0],
[1,0,0],
[0,1,0],
[0,0,1],
[-1,0,0],
[0,-1,0],
[0,0,-1],
[1,1,0],
[-1,1,0],
[-1,-1,0],
[1,-1,0],
[0,1,-1],
[0,1,1],
[0,-1,1],
[0,-1,-1],
[1,0,-1],
[-1,0,-1],
[-1,0,1],
[1,0,1],
[1,1,1],
[-1,1,1],
[1,-1,1],
[1,1,-1],
[-1,-1,-1],
[-1,-1,1],
[1,-1,-1],
[-1,1,-1],
[2,0,0],
[0,2,0],
[0,0,2],
[-2,0,0],
[0,-2,0],
[0,0,-2],
[2,2,0],
[-2,2,0],
[-2,-2,0],
[2,-2,0],
[0,2,-2],
[0,2,2],
[0,-2,2],
[0,-2,-2],
[2,0,-2],
[-2,0,-2],
[-2,0,2],
[2,0,2],
[2,2,2],
[-2,2,2],
[2,-2,2],
[2,2,-2],
[-2,-2,-2],
[-2,-2,2],
[2,-2,-2],
[-2,2,-2],
[3,0,0],
[0,3,0],
[0,0,3],
[-3,0,0],
[0,-3,0],
[0,0,-3] ] ) 

import numpy as np

pairList = []
for n,c in enumerate(C):
    pairList.append( sorted( (n, np.where( np.all( C== - C[n], axis = 1) )[0][0] ) ) )

pairList = np.unique( pairList  , axis = 0).flatten()

C = C[ pairList[ sorted( np.unique(pairList, return_index=True)[1] )  ] ]
cx,cy,cz= C.T
print( 'const int cx[NUM_OF_VEL] = { ' + ', '.join( '{:2d}'.format(x) for x in cx) + "};")
print( 'const int cy[NUM_OF_VEL] = { ' + ', '.join( '{:2d}'.format(x) for x in cy) + "};")
print( 'const int cz[NUM_OF_VEL] = { ' + ', '.join( '{:2d}'.format(x) for x in cz) + "};")

w = np.zeros( 59, dtype =np.float64)
w[0] = 0.09587891623775283272909445023188495617510262739235995610954304912304445365517319878804123312823232192;
for i in range(1,7): w[i] = 0.07310470821291483910941745565843276431330450303772922155521095094292168286597921449045544064733893514
for i in range(7,19): w[i] = 0.00346588971093380044024968482340080988496856481129172497341879120801011090308892581576297866102733265
for i in range(19,27): w[i] = 0.03661080820445153787374377062070467040982701208242066144446577453309397884381631060465021370053054559
for i in range(27,33): w[i] = 0.01592352322320595532135427342820581020467841425305806594860459987391070232076080272388080131254094406
for i in range(33,45) : w[i] = 0.002524808451050943939087549285728678043938383563209112997481736209858931176031162679906892857961001777
for i in range(45,53) : w[i] = 0.00007269686625151586346436623682506844944420728330856371497807347854142887028835193324688776081409868758
for i in range(53,59) : w[i] = 0.0007658794393468397060938785130819717826577889071787436568677554780749114270180559457876741989785392435

print( "const double w[NUM_OF_VEL] = ", ", ".join( ["{:.18e}".format(v)  for v in w ]) + "};")

f = np.array( [ sp.IndexedBase("f", shape  = (59,) )[i] for i in range(0,59) ] )

cdict = { 'x' : cx, 'y' : cy, 'z' : cz}

print( "m[1] = ( " + sp.ccode( np.dot( f , cx ) ) + " )/m[0];" )
print( "m[2] = ( " + sp.ccode( np.dot( f , cy ) ) + " )/m[0];" )
print( "m[3] = ( " + sp.ccode( np.dot( f , cz ) ) + " )/m[0];" )
print( "m[4] = ( " + sp.ccode( np.dot( f , cx*cx+cy*cy+cz*cz ) ) + " )/m[0];" )

mdict = { 'rho': 'm[0]' ,
'ux': 'm[1]',
'uy': 'm[2]',
'uz': 'm[3]',
'theta' : 'm[4]',
'Bx' : 'm[5]',
'By' : 'm[6]',
'Bz' : 'm[7]',
'tau_xx': 'm[8]',
'tau_xy': 'm[9]',
'tau_xz': 'm[10]',
'tau_yy': 'm[11]',
'tau_yz': 'm[12]',
'tau_zz': 'm[13]',
'qx': 'm[14]',
'qy': 'm[15]',
'qz': 'm[16]'
}

for a,b in list( itertools.combinations_with_replacement( ("x","y","z") , 2  ) ):
    print( "// tau_"  + "".join( (a,b) ) ) 
    print( mdict["tau_"  + "".join( (a,b) )] + " = "  + sp.ccode( np.dot( f , cdict[a]*cdict[b] ) ) + " - (m[0]*{0}*{1})".format( mdict[ "u"+a ]  , mdict[ "u"+ b ] ) + "+ ({0}*{1})".format( mdict[ "B"+a ]  , mdict[ "B"+ b ] )  +  ( "- ({0}*{1})".format( mdict["rho"] , mdict["theta"] ) + " - (0.5*B2) ;" if (a==b) else ";" ) )

tau = 0.3;
wc =  0.2;

hb = np.array( [-0.11, 0.35, 0.27] )
hb = hb/ np.linalg.norm(hb)
Q = (1/(2*tau) + 1 )*np.eye(3) + 0.5*wc*np.array( [ [    0     ,  hb[2]  , - hb[1]  ],
                       [  -hb[2]  ,    0    , + hb[0]  ],
                       [  +hb[1]  , -hb[0]  ,   0      ] ] )


qtilde = np.array( [1.27416, -4.38633, -16.8902 ] )

np.linalg.inv(Q) @ qtilde
