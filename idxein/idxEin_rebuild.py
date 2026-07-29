#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:29:38 2026

@author: diogo
"""

import sympy as sp
import numpy as np
from sympy import Symbol, Derivative, latex
from collections import Counter
from sympy.core.sorting import default_sort_key
import sympy as sp
from sympy.tensor.indexed import IndexException
from sympy.core.singleton import S
from itertools import combinations_with_replacement, permutations, product

sp.init_printing()

class IdxEin(sp.tensor.indexed.Idx):

    def __new__(cls, label , range=None, **kw_args):

      if (range == None):
          range = (1,3)

      obj = super().__new__(cls, label, range, **kw_args)

      strlabel = str(label)
      sep = strlabel.find("_")
      if ( sep != -1 ):
          obj.number = int(strlabel[sep+2:-1])
          obj.generator_label = strlabel[:sep]
          obj.range = range
      else:
          obj.number = None
          obj.generator_label = None
          obj.range = range          
      return obj

    def _latex(self, printer):
        # Customize the LaTeX representation of the Derivative object here
        return f'{self.name}'

    def compatible(self, other):
        return (
            isinstance(other, IdxEin)
            and self.lower == other.lower
            and self.upper == other.upper
        )

    def dimension(self):
        return self.upper - self.lower + 1

class IdxEinGenerator:

    def __init__(self, label, index_range=None):

        self.label = label
        self.index_range = index_range

    def _make_index(self, n):
        label = rf"{self.label}_{{{n}}}"

        if self.index_range is None:
            return IdxEin(label )

        return IdxEin(label, range=self.index_range)

    def __getitem__(self, key):
        # Caso simples: alpha[1]
        if isinstance(key, int):
            return self._make_index(key)

        # Caso com slice: alpha[1:4]
        if isinstance(key, slice):
            if key.stop is None:
                raise ValueError(
                    "O limite final do slice deve ser especificado."
                )

            start = 1 if key.start is None else key.start
            step = 1 if key.step is None else key.step

            if not all(
                isinstance(value, int)
                for value in (start, key.stop, step)
            ):
                raise TypeError(
                    "Os limites e o passo devem ser inteiros."
                )

            if step == 0:
                raise ValueError("O passo do slice não pode ser zero.")

            # O stop é inclusivo.
            inclusive_stop = key.stop + (1 if step > 0 else -1)

            return tuple(
                self._make_index(n)
                for n in range(start, inclusive_stop, step)
            )

        raise TypeError(
            "O gerador aceita somente inteiros ou slices."
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"{self.label!r}, index_range={self.index_range!r})"
        )

class IndexedEin(sp.Indexed):

    # __new__ e demais métodos...

    def _rebuild(self, base, indices):
        return type(self)(
            base,
            *indices,
            symmetric_groups=self.symmetric_groups,
            antisymmetric_groups=self.antisymmetric_groups,
        )

    def _eval_subs(self, old, new):
        new_base = self.base.subs(old, new)

        new_indices = tuple(
            index.subs(old, new)
            for index in self.indices
        )

        if (
            new_base == self.base
            and new_indices == self.indices
        ):
            return None

        return self._rebuild(
            new_base,
            new_indices,
        )

    def __new__(
        cls,
        base,
        *indices,
        symmetric_groups=(),
        antisymmetric_groups=(),
        **kwargs,
    ):
        normalized_indices = list(indices)
        sign = S.One

        # Primeiro, normalizamos os grupos simétricos.
        for group in symmetric_groups:
            values = [
                normalized_indices[position]
                for position in group
            ]

            ordered_values = sorted(
                values,
                key=default_sort_key,
            )

            for position, value in zip(
                group,
                ordered_values,
            ):
                normalized_indices[position] = value

        # Depois, normalizamos os grupos antissimétricos.
        for group in antisymmetric_groups:
            values = [
                normalized_indices[position]
                for position in group
            ]

            # Um tensor alternante é zero quando dois índices
            # do mesmo grupo coincidem.
            if len(set(values)) != len(values):
                return S.Zero

            # Posições originais dos elementos após a ordenação.
            permutation = sorted(
                range(len(values)),
                key=lambda position: default_sort_key(
                    values[position]
                ),
            )

            sign *= permutation_sign(permutation)

            ordered_values = [
                values[position]
                for position in permutation
            ]

            for position, value in zip(
                group,
                ordered_values,
            ):
                normalized_indices[position] = value

        component = super().__new__(
            cls,
            base,
            *normalized_indices,
            **kwargs,
        )

        component.symmetric_groups = symmetric_groups
        component.antisymmetric_groups = antisymmetric_groups

        return sign * component

    def _eval_derivative(self, wrt):
        if (
            isinstance(wrt, IndexedEin)
            and wrt.base == self.base
        ):
            if len(self.indices) != len(wrt.indices):
                raise IndexException(
                    "Different # of indices: "
                    "d({!s})/d({!s})".format(self, wrt)
                )

            result = S.One

            for index1, index2 in zip(
                self.indices,
                wrt.indices,
            ):
                result *= KroneckerDeltaEin(
                    index1,
                    index2,
                )

            return result

        return super()._eval_derivative(wrt)

class IndexedBaseEin( sp.IndexedBase):

    def __getitem__(self, indices, symmetric = False, antisymmetric = False, **kw_args):
      item = super().__getitem__(indices, **kw_args)
      return IndexedEin(self, *item.indices, **kw_args)

class KroneckerDeltaEin(IndexedEin):

    def __new__(cls, *args, **kwargs):

        if len(args) == 2:
            # Chamada pública:
            # KroneckerDeltaEin(i, j)
            i, j = args

        elif len(args) == 3:
            # Reconstrução do SymPy:
            # KroneckerDeltaEin(base, i, j)
            base, i, j = args

        else:
            raise TypeError(
                "KroneckerDeltaEin espera (i, j) "
                "ou (base, i, j)."
            )

        if isinstance(i, IdxEin) and isinstance(j, IdxEin):
            if not i.compatible(j):
                raise ValueError(
                    f"Índices com intervalos incompatíveis: "
                    f"{i}={i.lower, i.upper} e "
                    f"{j}={j.lower, j.upper}"
                )

            if i == j:
                return i.dimension()

        if isinstance(i, (int, sp.Integer)) and isinstance(
            j, (int, sp.Integer)
        ):
            return S.One if i == j else S.Zero

        return super().__new__(
            cls,
            IndexedBaseEin(r"\delta"),
            i,
            j,
            symmetric_groups=((0, 1),),
            **kwargs,
        )

    def _rebuild(self, base, indices):
    # O base é fixo e não faz parte da assinatura pública
    # de KroneckerDeltaEin.
        return type(self)(*indices)

def expandPow(a):
    a = sp.expand(a)
    
    if isinstance(a ,sp.Add):          
        return sp.Add( *[ expandPow(arg) for arg in a.args ] )
  
    elif isinstance(a ,sp.Mul):
        newargs = []
        for arg in a.args:
            if isinstance(arg ,sp.Mul):
                newargs +=  expandPow(arg).args
            elif isinstance(arg ,sp.Pow):
                b,e = arg.as_base_exp()
                if isinstance(b, IndexedEin):
                    newargs += expandPow(arg).args
                else:
                    newargs.append(arg)
            else:
                newargs.append(arg)
        return sp.Mul( *newargs , evaluate = False )
        
    elif isinstance(a, sp.Pow):
          b,e = a.as_base_exp()
          if isinstance(b, IndexedEin):
              return sp.Mul( *( e*[b]) , evaluate = False )
          else:
              return a
    else:
        return a

def getIdxEin(args):
    idxList = []
    for arg in args:
      if hasattr(arg, "indices"): idxList+= filter( lambda x : isinstance(x,IdxEin) ,arg.indices )
      idxList +=   getIdxEin(arg.args)
    return idxList

def normalizeIdxEin(
    exp,
    dummy_label = r"\beta",
):
    exp = sp.expand(exp)
    
    if isinstance(exp, sp.Add):
        return sp.Add(*[
            normalizeIdxEin(term, dummy_label =  dummy_label)
            for term in exp.args
        ])

    exp = expandPow(exp)
    sortedArgs = sorted( exp.args, key = lambda arg : hash( arg.base if isinstance(arg, IndexedEin)  else arg  )  )
    indices = getIdxEin( sortedArgs )
    counts = Counter(indices)   

    dummy_indices = { k : IdxEinGenerator(f"temp", index_range = k.range)[n] for n,(k,v) in enumerate(counts.items()) if v == 2 }
    used_indices  = [ k for k,v in counts.items() if v == 1 ]
   
    substitutions = {}
    for oldidx in dummy_indices.keys():       

        if (oldidx.generator_label != None):        
            generator_label = oldidx.generator_label
        else:
            generator_label = dummy_label
        
        n = 1        
        while (True):
            newidx = IdxEinGenerator(generator_label, index_range= oldidx.range )[n]
            if newidx in used_indices:
                n = n+1
            else:
                substitutions[ dummy_indices[oldidx] ] = newidx
                used_indices.append( newidx )
                break

    return expandPow( exp.subs( dummy_indices).subs( substitutions ) )

def simplifyEin(exp):
     expanded = expandPow( exp )

     if ( expanded.func == sp.Add ):
         return expandPow( sp.simplify( sp.Add( *( simplifyEin(arg) for arg in expanded.args) ) ) )

     countIndices = Counter( getIdxEin( expanded.args) )
     
     for n,arg in enumerate(expanded.args):                 
         if isinstance(arg, KroneckerDeltaEin):                     
             for n,idx in enumerate(arg.indices):
                 if countIndices[idx] == 2:
                    largs = list(expanded.args)
                    largs.remove(arg)
                    return  normalizeIdxEin( simplifyEin( sp.Mul(*largs).subs( { idx : arg.indices[(n+1)%2]}) ))
                
     return expanded

def HermiteTensor( x, indices ):
  if len(indices) == 0:  return S.One
  else:
    return ( - sp.diff(HermiteTensor(x, indices[:-1]),x[indices[-1]]) + x[indices[-1]]*HermiteTensor(x, indices[:-1]) ).expand()

def HermiteTensorPhysics( x, indices ):
  if len(indices) == 0:  return S.One
  else:
    return ( - sp.diff(HermiteTensorPhysics(x, indices[:-1]),x[indices[-1]]) + 2*x[indices[-1]]*HermiteTensorPhysics(x, indices[:-1]) ).expand()

def isotropicTensor( *args ):

    n = len(args)

    if (n == 0): return S.One
    if (n%2 == 1): return S.Zero

    tensor = S.Zero
    for i in range(1,n):
        tensor += KroneckerDeltaEin(args[0], args[i]) * isotropicTensor( *(args[1:i] + args[i+1:])  )

    return tensor

def computeMomentPhysics(exp, variable):

    term = expandPow( exp )
    if ( term.func == sp.Add ):
        return sp.Add( *(  computeMomentPhysics(arg, variable) for arg in term.args) )

    if (term.func == sp.Mul ):
        replace = [ arg for arg in term.args if isinstance(arg, IndexedEin) if (arg.base == variable) if isinstance(arg.indices[0],IdxEin)  ]
        keep    = [ arg for arg in term.args if not( arg in replace ) ]
        indices = [ arg.indices[0]  for arg in replace ]
        keep.append( isotropicTensor( *indices  ) / sp.Pow(2, sp.Rational( len(indices), 2 ) ) )
        return expandPow( simplifyEin( sp.Mul( *keep ) ))

    if (term.func == IndexedEin ):
        if (term.base == variable):
            return S.Zero

    return term

def dot(a, b, format="latex"):

    if format == "c":
        if a == b:
            return sp.Symbol(f"{a.name}{a.name}".replace("\\", ""))
        else:
            return sp.Symbol(f"{a.name}{b.name}".replace("\\", ""))

    elif format == "latex":
        if a != b:
            return sp.Symbol(
                "(\\boldsymbol{{{}}}\\cdot\\boldsymbol{{{}}})".format(
                    a.name, b.name
                )
            )
        else:
            return sp.Symbol(
                "(\\boldsymbol{{{}}}\\cdot\\boldsymbol{{{}}})".format(
                    a.name, a.name
                )
            )

def checkIdxEin(
    exp,
):
    exp = sp.expand(exp)

    if isinstance(exp, sp.Add):
        freeindexes = checkIdxEin(exp.args[0])
        for term in exp.args:
            if (checkIdxEin(term) != freeindexes):
              raise ValueError("IdxEin: Incompatible free indexes")

        return freeindexes
    term = expandPow(exp)


    indices = getIdxEin(term.args)
    counts = Counter(indices)

    dummy_indices = []
    free_indices = []
    gen_indices = {}

    for index in counts:
        if counts[index] == 2 and (index not in dummy_indices):  dummy_indices.append(index)
        elif counts[index] == 1 and (index not in free_indices): free_indices.append(index)
        else:
            raise ValueError("IdxEin: 3 repeated indexes")

    return set(free_indices)

def einsteinToProduct(term, index=None, format = "latex"):

    if term.func == sp.Add:
        return sp.Add(*(einsteinToProduct(arg, index, format = format) for arg in term.args))

    if term.func != sp.Mul:
        return term

    indices = set()

    for arg in term.args:
        if isinstance(arg, IndexedEin):
            for idx in arg.indices:
                if isinstance(idx, IdxEin):
                    if index is None or idx.compatible(index):
                        indices.add(idx)

    newargs = []
    used = set()

    for idx in indices:
        pos = []

        for n, arg in enumerate(term.args):
            if isinstance(arg, IndexedEin) and idx in arg.indices:
                pos.append(n)

        if len(pos) == 2:
            name0 = term.args[pos[0]].base
            name1 = term.args[pos[1]].base
            newargs.append(dot(name0, name1, format = format))
            used.update(pos)

    for n, arg in enumerate(term.args):
        if n not in used:
            newargs.append(arg)

    return sp.Mul(*newargs)