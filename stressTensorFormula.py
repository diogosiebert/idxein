#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:49:37 2023

@author: diogo
"""

import sympy as sp

sp.init_printing()


tau = sp.Symbol(r"\tau")
wc  = sp.Symbol(r"\omega_c")
b   = sp.IndexedBase(r"b", shape = (3,))
lbd = sp.IndexedBase(r"\lambda")
m = sp.IndexedBase("m", shape = (11,) )
midx = { (0,0) : 0 , (0,1) : 1, (1,0) : 1, (0,2) : 2, (2,0) : 2, (1,1) : 3,  (1,2) : 4, (2,1) : 4,  (2,2) : 5}
lbd1 = sp.Symbol(r"lbd1")
lbd2 = sp.Symbol(r"lbd2")

As = sp.Symbol(r"A_s")
Bs = sp.Symbol(r"B_s")
Cs = sp.Symbol(r"C_s")
Ds = sp.Symbol(r"D_s")

r = sp.IndexedBase("R", shape = (6) )
RM = sp.Matrix( [[ r[0] , r[1] , r[2] ] , 
                 [ r[1] , r[3] , r[4] ] ,
                 [-r[2] ,-r[4] , r[5] ] ] )  

K = sp.Matrix([[0, 0, b[0]],
              [0, 0, b[1]],
              [-b[0], -b[1], 0]])
R = sp.eye(3) + K + (1-b[2]) / (b[0]**2 + b[1]**2) * (K @ K)

for i,j in [ [0,0] , [0,1] , [0,2] , [1,1] , [1,2] , [2,2] ]:
    print( sp.ccode( RM[i,j] ) , " = " , sp.ccode( R[i,j] ) + ";" )
    
print()

def computeSystemSym( b ):
    M = sp.Matrix.zeros( 6 )
    for al,bt in [ (0,0), (0,1), (0,2) , (1,1), (1,2), (2,2) ]:
        M[ midx[ (al,bt) ] , midx[ (al,bt) ] ] = 1
        for gm in range(3):
            for nu in range(3):
                 M[ midx[ (al,bt) ] , midx[ (gm,bt) ] ] += lbd2 * b[nu] * sp.LeviCivita(gm,nu,al) 
            for nu in range(3):
                 M[ midx[ (al,bt) ] , midx[ (gm,al) ] ] += lbd2 * b[nu] * sp.LeviCivita(gm,nu,bt) 
    return M


InvMatrix = computeSystemSym(b).subs( { b[0] : 0, b[1] : 0, b[2]: 1} ).inv() 
tildeS = sp.Matrix( [m[i] for i in range(6) ] )
S = ( InvMatrix@tildeS ).subs( { 1/ (4* lbd2**2 + 1) : As,  1/ (lbd2**2 + 1) : Bs ,  2 *lbd2**2: Cs}  )

const = { 1/ (4* lbd2**2 + 1) : As,  1/ (lbd2**2 + 1) : Bs ,  2 *lbd2**2: Cs} 

  

sgm = sp.Matrix( [ [ m[ 5 + midx[ (i,j) ] ] for j in range(3) ] for i in range(3) ] )

for n,(i,j) in enumerate([ [0,0] , [0,1] , [0,2] , [1,1], [1,2] , [2,2] ]):
    print( "sR[{}] = ".format(n) , sp.ccode( (RM.T @ sgm @ RM)[i,j] ) + ";" )
    
print("\n")

m = sp.IndexedBase("sR", shape = (11,) )

tildeS = sp.Matrix( [m[i] for i in range(6) ] )
S = ( InvMatrix@tildeS ).subs( { 1/ (4* lbd2**2 + 1) : As,  1/ (lbd2**2 + 1) : Bs ,  2 *lbd2**2: Cs}  )

for n,s in enumerate(S):
    print( "s[{}] = ".format(n) + sp.ccode( sp.factor( lbd1 * s )  ) + ";") 

print("\n")


m = sp.IndexedBase("s", shape = (11,) )
sgm = sp.Matrix( [ [ m[ midx[ (i,j) ] ] for j in range(3) ] for i in range(3) ] )

for n,(i,j) in enumerate([ [0,0] , [0,1] , [0,2] , [1,1], [1,2] , [2,2] ]):
    print( "m[{}] = ".format(n+5) , sp.ccode( (RM @ sgm @ RM.T)[i,j] ) + ";" )
    
print("\n")


# sp.init_printing()
# delta = sp.Symbol( "\\delta", real = True)
# tau = sp.Symbol( "\\tau", real = True)
# wc = sp.Symbol("\\omega_c", positive = True ) 
# hb = IndexedBase("b")
# s = IndexedBase("\\sigma")
# sr = IndexedBase("\\tilde \\sigma")
# st = IndexedBase("\\tilde \\sigma")
# k = IdxEin("\\kappa", (1,3) )
# g = IdxEin("\\gamma", (1,3) )
# n = IdxEin("\\nu"   , (1,3) )
# b = IdxEin("\\beta" , (1,3))
# a = IdxEin("\\alpha", (1,3) )

# m = sp.IndexedBase("m")


# s_code = { s[1,1] : m[5] , s[1,2] : m[6] , s[1,3] : m[7] , s[2,2] : m[8] , s[2,3] : m[9] }
# subsym = { s[2,1] : s[1,2] , s[3,1] : s[1,3], s[3,2] : s[2,3] , s[3,3] : - s[1,1] - s[2,2] }
# # Eq = lambda x, y :  sp.Eq( ((1+delta/(2*tau)) * tau[ a, b ] - sp.summation( wc * delta / 2 * ( sp.LeviCivita( g,  n, a) *hb[n] *  tau[ b, g ] + sp.LeviCivita( g,  n, b) *hb[n] *  tau[ a, g ] ) , g , n )).xreplace( {a : x, b: y} ) , tst[x,y] )
# # EqSet = [ Eq(1,1), Eq(1,2), Eq(1,3).subs(subsym), Eq(2,2), Eq(2,3), Eq(3,3).subs( subsym ) ]

# LHS = lambda x,y : ((1+delta/(2*tau)) * s[ a, b ]+  wc * delta / 2 * sp.summation( hb[n] *( sp.LeviCivita( g,  n, a) *  s[ g, b ] + sp.LeviCivita( g,  n, b) *  s[ g, a ] ) , g , n )).xreplace( {a : x, b: y} )

# A = sp.Matrix( [ LHS(1,1).subs( subsym  ), LHS(1,2).subs( subsym  ), LHS(2,2).subs( subsym  ) , LHS(2,3).subs( subsym  ),  LHS(1,3).subs( subsym  )  ] )
# S =  sp.Matrix( [ s[1,1], s[1,2] , s[1,3], s[2,2] , s[2,3] ] )

# S = sp.Matrix( [  [  s[ sorted([i,j]) ]  for j in range(1,4) ] for i in range(1,4) ] )
# At = A.subs( { hb[1] : 0, hb[2] : 0, hb[3] : 1 } ).subs( { s : sr })
# Sr = R.T @ S @ R

# At.subs( { sr[i,j] : Sr[i-1,j-1] for i in range(1,4) for j in range(1,4) } )

# bc = sp.Symbol("\\beta_c")

# Tsys = A.jacobian( S )
# TsysRot = Tsys.subs( { hb[1] : 0 , hb[2] : 0 , hb[3] : 1 , delta : 1 }  )
# TRotInv = TsysRot.subs( { wc : 2*bc /lbd  , 1+1/(2*tau) : 1/lbd  } ).inv() 

# TRotInvSub = sp.simplify( TRotInv ).subs( { lbd /(bc**2 + 1) : sp.Symbol("C_s") , lbd /(4*bc**2 + 1) : sp.Symbol("A_s") , 2*bc**2 : sp.Symbol("B_s")  } )

# for lhs,rhs in zip( S.subs( s_code ) , ( TRotInvSub @ S ).subs( s_code ) ):
#     print( "$$", sp.latex(lhs) ,"=",  sp.latex(rhs) , "$$")


# K = sp.Matrix( [  [ 0     ,   0   ,-hb[1] ] , 
#                   [ 0     ,   0   , hb[2] ] , 
#                   [ hb[1], -hb[2] ,   0 ] ])

# R = sp.eye(3) - sp.sqrt( hb[1]**2 +  hb[2]**2) * K + (1-hb[3]) * (K @ K)
 

# S = sp.Matrix( [  [  s[i,j] for j in range(1,4) ] for i in range(1,4) ] )

# #
# # lbd = (2 * ttau) / ( 2*ttau - 1 )
# #
# # simDict = {}

# # for k in range(1,4):
# #     simDict[ wc * hb[k] ] =  2*lbd*hb[k]  
# # simDict[ 1 + 1/(2*tau) ] =  lbd

# # Tsys = A.jacobian( ( s[1,1], s[1,2] , s[2,2], s[2,3],  s[1,3]  ) ).subs( { delta : 1 }).subs( simDict ) / lbd 


# # Tmod = Tsys.subs( { delta * wc * hb[1]: hb[1]  ,  delta * wc * hb[2] : hb[2] ,  delta * wc * hb[3] : lbda hb[3],   delta / (2* ttau ) + 1 :   } )
# # TmodInv = Tmod.inv()

# # Tinv = TmodInv.subs( { hb[1] : delta * wc * hb[1]  ,  hb[2]: delta * wc * hb[2] ,  hb[3] : delta * wc * hb[3] ,   sp.Symbol( "\\lambda" ) : delta / (2* ttau ) + 1 } )

# # Tfull = sp.Matrix( [ LHS(i,j) for i in range(1,4) for j in range(1,4) ] ).jacobian( [ s[i,j] for i in range(1,4) for j in range(1,4) ] )
# # TfullSub = Tfull.subs( { delta : 1 }).subs( simDict ) / lbd