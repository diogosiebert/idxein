#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:33:14 2023

@author: diogo
"""

import sympy as sp
import numpy as np
from sympy import Symbol, Derivative, latex
from collections import Counter

import sympy as sp
from sympy.tensor.indexed import IndexException
from sympy.core.singleton import S
from itertools import combinations_with_replacement

def unroll(a):
  newargs = []

  if isinstance(a , D):
      return D( unroll( a.expr ) , *a.variables )     

  if isinstance(a ,sp.Add):
      return sp.Add( *[ unroll(arg) for arg in a.args ] )

  if isinstance(a ,sp.Mul):
    for arg in a.args:
        unrolled = unroll(arg)
        if ( isinstance( unrolled, sp.Mul) ):
            newargs += list( unrolled.args )
        else:
            newargs.append(unrolled)

  elif isinstance(a,sp.Pow):
      b,e = a.as_base_exp()
      if ( e.is_Integer ) and ( e > 0):
        newargs +=  e*[b]
      else:
        return a
  else:
      return a

  return sp.Mul( *newargs , evaluate = False )

class IdxEin(sp.tensor.indexed.Idx):

  def __new__(cls, label, range=None, **kw_args):
    if (range == None):
        range = (1,3)
    return super().__new__(cls,label, range=range, **kw_args)

  def _latex(self, printer):
      # Customize the LaTeX representation of the Derivative object here
      return f'{self.name}'

  def _subs(self, old, new ):
      return self.func( *( arg._subs( old, new ) for arg in self.args ) )

  def compatible(self, other):
      return (other.lower == self.lower) and (other.upper == self.upper)

class KroneckerDelta( sp.KroneckerDelta ):

   @classmethod
   def eval(cls, i, j, delta_range=None):
      if (not isinstance(i, IdxEin)):        
        return super(KroneckerDelta, cls).eval(i,j, delta_range)
      else:
        if ( (i==j) ): return i.upper - j.lower + 1
        mylist = [i,j]
        mylist.sort( key = str)
        return KroneckerDelta( *mylist , evaluate = False )

   def _eval_power(self, expt):

      if any(isinstance( i , IdxEin) for i in self.indices):
        pass
      else:
        return super()._eval_power(expt)

   def doit(self, **hints):
      if ( self.indices[0] == self.indices[1] ): return S.One
      else: return self.eval( *self.indices )

   def isEinstein(self):
       if not (isinstance(self.indices[0],IdxEin) and isinstance(self.indices[0],IdxEin)):
           return False
       if not (self.indices[0].lower == self.indices[1].lower):
           return False
       if not (self.indices[0].upper == self.indices[1].upper):
           return False
       return True

class Indexed(sp.Indexed):

    def _eval_derivative(self, wrt):

        if isinstance(wrt, Indexed) and wrt.base == self.base:
            if len(self.indices) != len(wrt.indices):
                msg = "Different # of indices: d({!s})/d({!s})".format(self,
                                                                       wrt)
                raise IndexException(msg)
            result = S.One
            for index1, index2 in zip(self.indices, wrt.indices):
                result *= KroneckerDelta(index1, index2)
            return result

class IndexedBase( sp.IndexedBase):

    def __getitem__(self, indices, **kw_args):
      item = super().__getitem__(indices, **kw_args)
      return Indexed(self, *item.indices, **kw_args)

class D(Derivative):

    def __new__(cls, expr, *variables, **kwargs):
        
        if (isinstance(expr,D)):
            return D(expr.expr, *(variables + expr.variables), **kwargs)
                    
        if (isinstance(expr,sp.Matrix)):
            return sp.Matrix( [ D( x,  *variables, **kwargs) for x in expr ]  ).reshape( *expr.shape )

        return super().__new__(cls, expr, *variables, *kwargs)

    def _latex(self, printer):
        # Customize the LaTeX representation of the Derivative object here
        return ''.join( [ f'\\partial_{{ {latex( var.indices[0] if var.is_Indexed else var)} }}' for var in self.variables ] + [f'\\left( {latex(self.expr)} \\right)'])

    def subs(self, varDict ):
        return self.func( *( arg.subs( varDict ) for arg in self.args ) )

    def _subs(self, old, new ):
        return self.func( *( arg._subs( old, new ) for arg in self.args ) )

def getEinsteinIndices(args):
    idxList = []
    for arg in args:
      if hasattr(arg, "indices"): idxList+= filter( lambda x : isinstance(x,IdxEin) ,arg.indices )
      if isinstance( arg,  D):
          idxList +=  getEinsteinIndices( arg.variables)
          idxList +=  getEinsteinIndices( (arg.expr,) )
      else:
          idxList +=  getEinsteinIndices(arg.args)
    return idxList

def HermiteTensor( idx , x):
  if ( len(idx)==0):  return S.One
  else:
    return ( - sp.diff(HermiteTensor(idx[:-1],x),x[idx[-1] ] ) + x[idx[-1] ]*HermiteTensor( idx[:-1],x) ).expand()

def simplifyKronecker(exp, sumcancel = True):
    
     if ( isinstance(exp, sp.Matrix) ):
         return exp.subs( { x : simplifyKronecker(x,sumcancel) for x in exp} ) 
    
     expanded = exp.expand()

     if ( expanded.func == sp.Add ):
         
         newexp = sp.Add( *( simplifyKronecker(arg) for arg in expanded.args) )

         if (sumcancel) and (newexp.func == sp.Add):     
             
             sumIndices  = set()
            
             for arg in newexp.args:
                 indices = getEinsteinIndices( arg.args )
                 counts = Counter( indices )
                 sumIndices = sumIndices.union( [element for element, count in counts.items() if count == 2] )           
                
             sumIndices = sorted( list( sumIndices) , key = lambda x: x.name )
            
             newargs =  []
             
             for arg in newexp.args:
                 indices = getEinsteinIndices( arg.args )
                 counts = Counter( indices )
                 argIndices = [element for element, count in counts.items() if count == 2]
                 newargs.append( arg.xreplace(  { old : sumIndices[n] for n, old in enumerate( argIndices)  } ) )
                     
             return sp.Add( *newargs )
         else:
             return newexp

     unrolled = unroll(expanded)
     for n,arg in enumerate(unrolled.args):
        if isinstance(arg, KroneckerDelta):
            otherArgs = unrolled.args[:n] + unrolled.args[n+1:]
            otherIdx  = getEinsteinIndices(otherArgs)
            if arg.indices[0] in otherIdx:
                return simplifyKronecker( sp.Mul(*otherArgs).xreplace( { arg.indices[0] : arg.indices[1]}) )
            if arg.indices[1] in otherIdx:
                return simplifyKronecker( sp.Mul(*otherArgs).xreplace( { arg.indices[1] : arg.indices[0]}) )

     return unrolled


def isotropicTensor( *args ):
    
    n = len(args)
    
    if (n == 0): return S.One
    if (n%2 == 1): return S.Zero
    
    tensor = S.Zero
    for i in range(1,n):
        tensor += KroneckerDelta(args[0], args[i]) * isotropicTensor( *(args[1:i] + args[i+1:])  )
    
    return tensor
        
def computeMoment(exp, variable, index):
    
    term = exp.expand()    
    if ( term.func == sp.Add ):
        return  simplifyKronecker( sp.Add( *(  computeMoment(arg, variable, index) for arg in term.args) ) )
    
    term = unroll(exp)
    if (term.func == sp.Mul ):
        replace = [ arg for arg in term.args if isinstance(arg, Indexed) if (arg.base == variable) if isinstance(arg.indices[0],IdxEin) if arg.indices[0].compatible(index) ]
        keep    = [ arg for arg in term.args if not( arg in replace ) ]
        keep.append( isotropicTensor( *[ arg.indices[0]  for arg in replace ] ) )
        return unroll( sp.Mul( *keep ) )
    else:
        return computeMoment( sp.Mul( term , S.One , evaluate=False) , variable, index)
    
def getIndexed( x ):    
    if ( isinstance(x,Indexed) ):
        return { x.base }
    elif (x.args != None):
        result = set()
        for arg in x.args:
            base = getIndexed(arg)
            if (base != None):
                result = result.union( base )
        return result    
    return None

def createIndex( symbol, myrange , n):
    if (myrange == (1,3) ): idxstr = "{}_{}".format( symbol, n )    
    if (myrange == (1,D) ): idxstr = "\\underline{{{}}}_{}".format( symbol, n )
    if (myrange == (D+1,3) ): idxstr = "\\overline{{{}}}_{}".format( symbol, n )
    return IdxEin( idxstr, myrange)

greek = [r"\alpha",r"\beta",r"\gamma",r"\eta",r"\nu",r"\mu",r"\xi"]    

def replaceIndeces( exp , idxDict = None):

    if (exp.func == sp.Add ):
        idxDict  = { t: greek[n] for n,t in enumerate( [ tuple(sorted(x, key = lambda x : x.name )) for x in combinations_with_replacement( getIndexed( exp ), 2) ] ) }
        return sp.Add( *[ replaceIndeces(arg, idxDict = idxDict) for arg in exp.args] )
        
    if (exp.func == sp.Mul ):
        idxCount = { idx : 0 for idx in idxDict.values() } 
        indices = np.array( [ arg.indices[0] if isinstance(arg,Indexed) else None for arg in exp.args ] )
        bases   = np.array( [ arg.base   if isinstance(arg,Indexed) else None for arg in exp.args ] )
        
        subsDict = {}
        for idx in set( indices ).difference( {None} ):
            pair = tuple( sorted( bases[ np.where( indices == idx )[0] ] ,  key =  lambda x : x.name ) )
            if (pair in idxDict.keys() ):                                 
                idxsymbol = idxDict[ tuple( sorted( bases[ np.where( indices == idx )[0] ] ,  key =  lambda x : x.name ) ) ]
                idxRange = ( idx.lower, idx.upper )
                idxCount[idxsymbol] += 1
                newIdx = createIndex( idxsymbol,  idxRange, idxCount[idxsymbol] )
                subsDict[idx] = newIdx
            
        return unroll(  exp.xreplace( subsDict ) )
    
    return exp

def newReplaceIndeces( exp , idxDict = None, human = True):

    if (exp.func == sp.Add ):
        idxDict  = { t: '{}replace'.format(n) for n,t in enumerate( [ tuple(sorted(x, key = lambda x : x.name )) for x in combinations_with_replacement( getIndexed( exp ), 2) ] ) }
        newexp = sp.Add( *[ replaceIndeces(arg, idxDict = idxDict, human = False) for arg in exp.args] )        

    elif (exp.func == sp.Mul ):
        idxCount = { idx : 0 for idx in idxDict.values() } 
        indices = [ k for x in exp.free_symbols if x.is_Indexed for k in x.indices if isinstance(k,IdxEin) ]               
        bases   = [ x.base for x in exp.free_symbols if x.is_Indexed for k in x.indices if isinstance(k,IdxEin) ]               
        replace = [ idx for idx, num in Counter( indices ).items() if num == 2 ]
        
        tempIdx = [ ]
        for idx in replace:
            pair = tuple( sorted( ( bases[n] for n,symbol in enumerate(indices) if symbol == idx ) , key = lambda x : x.name )  )
            idxCount[ idxDict[ pair  ]  ] += 1
            tempIdx.append( IdxEin( idxDict[ pair  ] + "_{}".format( idxCount[ idxDict[ pair  ]  ] ) ) )          
        newexp = unroll(  exp.xreplace( { old: new for old,new in zip(replace,tempIdx) }  ) )
                               
    if (human):
        nonHumanIndices = { k for x in newexp.free_symbols if x.is_Indexed for k in x.indices if isinstance(k,IdxEin) if k.name.find("replace") > 0 }                   
        allEinIndices      = [ k for x in newexp.free_symbols if x.is_Indexed for k in x.indices if isinstance(k,IdxEin) ]                   
        subs = { }
        n = 0
        for idx in nonHumanIndices:
            while True:
                n += 1
                newidx = IdxEin("\\alpha_{}".format(n))
                if newidx not in allEinIndices: break
            subs[ idx ] = newidx
        return newexp.xreplace( subs )
    else:          
        return newexp
    
def simplifyD( exp , constants = () ):

    if isinstance(exp,sp.Matrix):
        return sp.Matrix( [ simplifyD(x , constants = constants) for x in exp ] ).reshape( *exp.shape )  

    if (exp.func == sp.Mul):
        return sp.Mul( *[simplifyD(arg) for arg in exp.args ] )

    if (exp.func == D):

        if hasattr( exp.expr, "is_Number") :
            if exp.expr.is_Number :
                return S.Zero
        
        if (exp.expr.func == sp.Mul):
            terms = unroll(exp.expr)            
            const = [ arg for arg in terms.args if ( (arg in constants) or isinstance(arg, KroneckerDelta) or (arg.is_Number)) ]
            funct = [ arg for arg in terms.args if not arg in const ]
            return sp.Mul( *( sp.Mul( *const ) ,  D(sp.Mul( *funct ), *exp.variables ) ) )                    

        if (exp.expr.func == sp.Add):
            return sp.Add( *[ simplifyD( D( arg  , *exp.variables ) , constants =constants )  for arg in exp.expr.args ] )        

    return exp

def splitSum( term, index, var = None):
    
    exp = sp.expand(term)
    if (exp.func == sp.Add ):
         return sp.Add( *(  splitSum(arg, index, var) for arg in exp.args) )
    
    if (exp.func == sp.Mul ):
        exp = unroll(exp)
        replace = [ arg for arg in exp.args if (isinstance(arg,Indexed) ) if ( arg.indices[0].compatible(index) )]
        keep    = [ arg for arg in exp.args if not( arg in replace ) ]
        indices = [ arg.indices[0]  for arg in replace ]
        
        for idx in set(indices):
            _idx = IdxEin( "\\underline{{{}}}_{}".format(* idx.name.split("_") ) , range = (1,D) )
            idx_ = IdxEin( "\\overline{{{}}}_{}".format(* idx.name.split("_") ) , range = (D+1,3) )
            terms = [ arg for arg in replace if arg.indices[0] == idx ] 
            if (len(terms) != 2): raise ValueError("Something Wrong with the Einstein Convention" )
            else:
                if (var == None) or (var in [ c.base for c in terms ]):
                    keep.append( terms[0].xreplace( { idx : _idx } ) * terms[1].xreplace( { idx : _idx } ) +  terms[0].xreplace( { idx : idx_ } ) * terms[1].xreplace( { idx : idx_ } ) ) 
                else:
                    keep += terms
        
        return sp.expand( sp.Mul( *keep )  ) 

    return sp.Mul( term , S.One , evaluate=False)  

def simplifyByPermutation( exp, tensor):
    subdict = { x : x.base[ sorted( x.indices , key = lambda i : i.name ) ]  for x in exp.free_symbols if x.is_Indexed if x.base == tensor }
    return exp.subs( subdict )

def simplifyDeviatoric( exp, tensor ):
    subdict = { x : S.Zero  for x in exp.free_symbols if x.is_Indexed if x.base == tensor if len(x.indices) == 2 if x.indices[0] == x.indices[1]}
    return exp.subs( subdict )