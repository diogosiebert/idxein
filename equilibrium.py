#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 13:34:13 2023

@author: diogo
"""

import sympy as sp
from idxein import *
import itertools 
sp.init_printing()
Dim = 3
a  = lambda n : IdxEin("\\alpha_{}".format(n), range=(1,Dim+1) )
b  = lambda n : IdxEin("\\beta_{}".format(n), range=(1,Dim+1) )
g  = lambda n : IdxEin("\\gamma_{}".format(n), range=(1,Dim+1) )
B = sp.IndexedBase("B")

def MaxwellStress( i , j ):
    return B[i]*B[j] - B[ b(1) ]*B[ b(1 )] /2  * KroneckerDelta( i , j)

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


def einsteinToDot( exp ):    
    if ( exp.func == sp.Add):
        return sp.Add( *[ einsteinToDot(t) for t in exp.args ] )
    
    if ( exp.func == sp.Mul):
    
        unrolled =  unroll( exp ) 
        indices =  list( filter( lambda x : isinstance(x,IdxEin), unrolled.free_symbols ) )
        newsymbols = []
        oldsymbols = []
        for k in indices:
            old = [ x for x in unrolled.args if isinstance(x,Indexed) and x.indices[0] == k ]
            newsymbols.append( sp.Symbol(  "".join( sorted( [ x.base.name  for x in old ] ) ) ) )    
            oldsymbols += old            
        
        return sp.Mul( *[ s for s in unrolled.args if s not in oldsymbols ] + newsymbols )

    return exp

Dim = 2 #sp.Symbol("D", integer = True)


rho, tt, cs = sp.symbols("\\rho, \\theta, c_s", real = True)
i = sp.Idx("i")
w = sp.IndexedBase("w")
u_o = IndexedBase("u_o")
u = IndexedBase("u")
xi_o = IndexedBase("\\xi_o")
x = IndexedBase("x")
t = sp.Symbol("t", real = True, positive = True)
c = IndexedBase("c")
e = IndexedBase("c")
cs = sp.Symbol("c_s") 
af = 1/cs
TT = sp.Symbol("\\Theta")

idx = lambda n : [ a(i) for i in range(1,n+1) ]

feq = S.Zero
for n in range(3):
    H = HermiteTensor( idx(n), xi_o )
    A = computeMoment( H.subs( { xi_o[k] : sp.sqrt(TT+1)*c[k] + u_o[k] for k in idx(n) } )  , c , a(1) )
    feq +=  A*H / sp.factorial(n)      

PSI = [ 
(5-xi_o[b(1)]*xi_o[b(1)]+xi_o[1]**2*(-7+xi_o[b(1)]*xi_o[b(1)]))/(4*sp.sqrt(2)) ,
 (-(xi_o[1]**2-8*xi_o[2]**2)*xi_o[b(1)]*xi_o[b(1)]-7*(-5+9*xi_o[2]**2+xi_o[3]**2))/(12*sp.sqrt(14)) ,
(35-70*xi_o[3]**2-(xi_o[1]**2+xi_o[2]**2-9*xi_o[3]**2)*xi_o[b(1)]*xi_o[b(1)])/(6*sp.sqrt(70)) ,
(xi_o[1]*xi_o[2]*(-7+xi_o[b(1)]*xi_o[b(1)]))/(sp.sqrt(14)) ,
(xi_o[1]*xi_o[3]*(-7+xi_o[b(1)]*xi_o[b(1)]))/(sp.sqrt(14)) ,
(xi_o[2]*xi_o[3]*(-7+xi_o[b(1)]*xi_o[b(1)]))/(sp.sqrt(14)) ]

A_PSI = [ computeMoment( psi.subs( { xi_o[k] : sp.sqrt(tt+1)*c[k] + u_o[k] for k in range(1,4) } ).subs( { xi_o[b(1)] :sp.sqrt(tt+1)*c[b(1)] + u_o[b(1)]  } ), c  ).xreplace( {b(1) :  b(2) } ) for psi in PSI ]      


#feq = simplifyKronecker( feq ).subs(  { xi_o[k]  : af * e[k] for k in idx(n) } ).subs(  { u_o[k]  : af * u[k] for k in idx(n) } )

feqc = einsteinToDot( feq )

feqM = S.Zero
for n in range(3):
    H = HermiteTensor( idx(n), c )
    A = computeMomentMagnetic( H  , c , a(1) )
    feqM +=  A*H / sp.factorial(n)
    
#feqM = simplifyKronecker( feqM ) .subs(  { B[k]  : af * B[k] for k in idx(n) } ).subs(  { c[k]  : af * c[k] for k in idx(n) } )    

feqMc = einsteinToDot( feqM )


def groupInPairs( mylist):

    lista = list( range( len(mylist) ) )
    pares = list(itertools.combinations(lista, 2))
    
    # Gera todas as combinações possíveis de pares sem repetições
    result = []
    for combinacao in itertools.combinations(pares, len(lista)//2):
        elementos = [item for sublista in combinacao for item in sublista]
        if len(set(elementos)) == len(lista):
            result.append(combinacao)
    
    filteredResult = []
    for case in result:
        group = tuple( (mylist[pair[0]],mylist[pair[1]]) for pair in case ) 
        if group not in filteredResult:
            filteredResult.append( group )
            
    return filteredResult

def detectEinstein(myexpand , i = 1 , force  = None):
    
    if (myexpand.func == sp.Add):
        return unroll( sp.expand( sp.Add( *[ detectEinstein(arg,i,force) for arg in myexpand.args ] ) ) )  
    
    if (myexpand.func == sp.Mul ):
        
        baseList = [ arg.base for arg in myexpand.args if arg.is_Indexed if arg.indices[0] == i ]
        
        possibilities = groupInPairs(baseList)
        
        if ( len(possibilities) == 1 ):
            group = possibilities[0]
        elif force  !=  None:
            group = possibilities[force]
        else:
            print("Term {} has {} possibilities of replacing by sum".format( sp.pretty( myexpand ), len(possibilities ) ) )
            return myexpand

        newterms = []
        oldterms = []             
                            
        for n,pair in enumerate(group):
             myidx = g(n+1)
             newterms.append( pair[0][myidx] * pair[1][myidx] - pair[0][1] * pair[1][1] - pair[0][2] * pair[1][2] -pair[0][3] * pair[1][3] + pair[0][i] * pair[1][i]   )
             oldterms+= [ pair[0][i], pair[1][i] ]

        keep = [ arg for arg in myexpand.args if arg not in oldterms ]
        return unroll(sp.Mul(*( keep + newterms)))
        
    return myexpand

def detectEinstein(myexpand , i = 1 , force  = None):
    
    if (myexpand.func == sp.Add):
        return newReplaceIndeces( unroll( sp.expand( sp.Add( *[ detectEinstein(arg,i,force) for arg in myexpand.args ] ) ) )  )
    
    if (myexpand.func == sp.Mul ):
        
        baseList = [ arg.base for arg in myexpand.args if arg.is_Indexed if arg.indices[0] == i ]
        
        possibilities = groupInPairs(baseList)
        
        if ( len(possibilities) == 1 ):
            group = possibilities[0]
        elif force  !=  None:
            group = possibilities[force]
        else:
            print("Term {} has {} possibilities of replacing by sum".format( sp.pretty( myexpand ), len(possibilities ) ) )
            return myexpand
        
        newterms = []
        oldterms = []             
                            
        for n,pair in enumerate(group):
             myidx = g(n+1)
             newterms.append( pair[0][myidx] * pair[1][myidx] - pair[0][1] * pair[1][1] - pair[0][2] * pair[1][2] -pair[0][3] * pair[1][3] + pair[0][i] * pair[1][i]   )
             oldterms+= [ pair[0][i], pair[1][i] ]

        keep = [ arg for arg in myexpand.args if arg not in oldterms ]
        return unroll(sp.Mul(*( keep + newterms)))
        
    return myexpand


exp4 = unroll( sp.expand( np.dot( A_PSI, PSI ) ) )
exp4_1 = detectEinstein( exp4, i = 1 , force = 0) 

done = sp.Add( *[ arg for arg in unroll(exp4_1).args if not any([ x.indices[0] in (1,2,3) for x in arg.args if x.is_Indexed ] ) ] )
todo = sp.Add( *[ arg for arg in unroll(exp4_1).args if any([ x.indices[0] in (1,2,3) for x in arg.args if x.is_Indexed ] ) ] )

todott0 = todo.subs({ tt: 0 })
todott1 = unroll( todo.diff( tt, 1).subs({ tt: 0 }) )

todott0xi4u4 = sp.Add( *[ exp for exp in todott0.args if np.all( np.sum( [ [arg.base == xi_o, arg.base == u_o] for arg in exp.args if arg.is_Indexed ] , axis =  0) == np.array([4,4]) ) ] )
todott0xi2u4 = sp.Add( *[ exp for exp in todott0.args if np.all( np.sum( [ [arg.base == xi_o, arg.base == u_o] for arg in exp.args if arg.is_Indexed ] , axis =  0) == np.array([2,4]) ) ] )
todott1xi4u2 = sp.Add( *[ exp for exp in todott1.args if np.all( np.sum( [ [arg.base == xi_o, arg.base == u_o] for arg in exp.args if arg.is_Indexed ] , axis =  0) == np.array([4,2]) ) ] )
todott1xi2u2 = sp.Add( *[ exp for exp in todott1.args if np.all( np.sum( [ [arg.base == xi_o, arg.base == u_o] for arg in exp.args if arg.is_Indexed ] , axis =  0) == np.array([2,2]) ) ] )

# myexpand = exp4
# force = 0

# for N in range(0,32):
#     newargs = []
#     force = [None] * 88
#     force[24], force[37], force[58],force[73] ,force[85] = [ int(c) for c in np.binary_repr( N ).zfill(5) ]
    
#     for n,arg in enumerate( myexpand.args) :  
#         newargs.append( detectEinstein(arg,1, force[n] )  )
        
#     exp4_1 = newReplaceIndeces( unroll( sp.expand( sp.Add( *newargs ) ) )  )
#     print( len( exp4_1.args ) )