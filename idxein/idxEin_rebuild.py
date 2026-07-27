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

    def __new__(cls, label,generator_label=None, number = None, range=None, **kw_args):
        
      if (range == None):
          range = (1,3)
          
      obj = super().__new__(cls,label, range=range, **kw_args)
      obj.number = number 
      obj.generator_label = generator_label
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
            return IdxEin(label, number = n )

        return IdxEin(label, generator_label=self.label, number = n,  range=self.index_range)

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
  newargs = []

  if isinstance(a ,sp.Add):
      return sp.Add( *[ expandPow(arg) for arg in a.args ] )

  if isinstance(a ,sp.Mul):
    for arg in a.args:
      if ( isinstance(arg ,(sp.Mul,sp.Pow) ) ):
        newargs += list( expandPow(arg).args )
      else:
        newargs.append(arg)
  elif isinstance(a,sp.Pow):
      b,e = a.as_base_exp()
      if ( e.is_Integer ):
        newargs +=  e*[b]
      else:
        return a
  else:
      return a

  return sp.Mul( *newargs , evaluate = False )

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
        return  expandPow( sp.Mul( *keep ) )
    
    if (term.func == IndexedEin ):        
        if (term.base == variable):            
            return S.Zero
        
    return term  

def getIdxEin(args):
    idxList = []
    for arg in args:
      if hasattr(arg, "indices"): idxList+= filter( lambda x : isinstance(x,IdxEin) ,arg.indices )
      idxList +=   getIdxEin(arg.args)
    return idxList

def simplifyEin(exp):
     expanded = expandPow( exp )

     if ( expanded.func == sp.Add ):
         return sp.Add( *( simplifyEin(arg) for arg in expanded.args) )

     for n,arg in enumerate(expanded.args):
        if isinstance(arg, KroneckerDeltaEin):
            otherArgs = expanded.args[:n] + expanded.args[n+1:]
            otherIdx  = getIdxEin(otherArgs)
            if arg.indices[0] in otherIdx:
                return simplifyEin( sp.Mul(*otherArgs).subs( { arg.indices[0] : arg.indices[1]}) )
            if arg.indices[1] in otherIdx:
                return simplifyEin( sp.Mul(*otherArgs).subs( { arg.indices[1] : arg.indices[0]}) )
     return expanded

# sp.init_printing()

D  = sp.symbols("D", integer = True, positive = True)
alpha = IdxEinGenerator(r"\alpha", index_range= (1,D) )
xi = IndexedBaseEin(r"\tilde \xi", real = True)
z  = IndexedBaseEin(r"z", real = True)
u  = IndexedBaseEin(r"\tilde u", real = True)
q  = IndexedBaseEin(r"\tilde q", real = True)
rho, tT = sp.symbols(r"\rho, \tilde{T}")
tt = sp.symbols(r"\theta")
Pr = sp.symbols("Pr")

f = S.Zero
N = 2
phiS = 1  + (1-Pr) *4 * q[ alpha[N+1] ] * z[ alpha[N+1] ] / (5 * rho * sp.sqrt( tT**3 ) ) * (2 * z[ alpha[N+2] ] *z[ alpha[N+2] ] - D - 2 )


for n in range(0,N+1):
      H = HermiteTensorPhysics( xi, alpha[1:n] )
      Hs = H.subs( { xi[ alpha[i+1] ] : z[alpha[i+1] ] * sp.sqrt(tT) + u[ alpha[i+1] ] for i in range(n)  } )
      a =  computeMomentPhysics(  Hs * phiS , z  )  / 2**n 
      f += 1/sp.factorial(n) * a * H


def normalizeIndex(
    exp,
    dummy_label=None,
):

    if dummy_label is None:
        dummy_label = r"\beta"

    exp = sp.expand(exp)

    if isinstance(exp, sp.Add):
        return sp.Add(*[
            normalizeIndex(term, dummy_label =  dummy_label)
            for term in exp.args
        ])

    term = expandPow(exp)

    indices = getIdxEin(term.args)
    counts = Counter(indices)

    dummy_indices = [
        index
        for index in counts
        if counts[index] == 2
    ]

    free_indices = [
        index
        for index in counts
        if counts[index] == 1
    ]

    substitutions = {}
    
    dummy_count = 1
    dummy = IdxEinGenerator(r"\beta")
    
    for oldidx in dummy_indices:
        
        if oldidx.number == None:            
            newidx = dummy[dummy_count]
            dummy_count += 1
        else:
            
            
        
        substitutions[oldidx] = newidx
                
    if not dummy_indices:
        return term



    # ---------------------------------------------------------
    # Índices numerados: agrupamento por família e intervalo.
    # ---------------------------------------------------------
    numbered_dummies = [
        index
        for index in dummy_indices
        if index.number is not None
    ]

    numbered_families = {}

    for index in numbered_dummies:
        family_key = (
            index.generator_label,
            index.lower,
            index.upper,
        )

        numbered_families.setdefault(
            family_key,
            [],
        ).append(index)

    for family_key, family_dummies in numbered_families.items():
        generator_label, lower, upper = family_key

        # Números que precisam permanecer reservados:
        # índices da mesma família que não serão renomeados.
        occupied_numbers = {
            index.number
            for index in counts
            if (
                index.number is not None
                and index.generator_label == generator_label
                and index.lower == lower
                and index.upper == upper
                and index not in family_dummies
            )
        }

        # Ordem canônica independente dos números originais.
        # Aqui, a ordem de aparição no termo define qual par
        # recebe primeiro o menor número disponível.
        ordered_dummies = []

        for index in indices:
            if (
                index in family_dummies
                and index not in ordered_dummies
            ):
                ordered_dummies.append(index)

        candidate_number = 1

        for old_index in ordered_dummies:
            while candidate_number in occupied_numbers:
                candidate_number += 1

            new_label = rf"{generator_label}_{{{candidate_number}}}"

            new_index = IdxEin(
                new_label,
                number=candidate_number,
                generator_label=generator_label,
                range=(lower, upper),
            )

            substitutions[old_index] = new_index
            occupied_numbers.add(candidate_number)
            candidate_number += 1

    # ---------------------------------------------------------
    # Índices não numerados: use o gerador dummy.
    # ---------------------------------------------------------
    unnumbered_dummies = [
        index
        for index in dummy_indices
        if index.number is None
    ]

    ordered_unnumbered = []

    for index in indices:
        if (
            index in unnumbered_dummies
            and index not in ordered_unnumbered
        ):
            ordered_unnumbered.append(index)

    # Evita colisão com índices já pertencentes à família dummy.
    occupied_dummy_numbers = {
        index.number
        for index in counts
        if (
            index.number is not None
            and index.generator_label == dummy.label
        )
    }

    candidate_number = 1

    for old_index in ordered_unnumbered:
        while candidate_number in occupied_dummy_numbers:
            candidate_number += 1

        generated_index = dummy[candidate_number]

        # O intervalo do índice substituto deve ser compatível
        # com o intervalo do índice original.
        if not old_index.compatible(generated_index):
            generated_index = IdxEin(
                rf"{dummy.label}_{{{candidate_number}}}",
                number=candidate_number,
                generator_label=dummy.label,
                range=(
                    old_index.lower,
                    old_index.upper,
                ),
            )

        substitutions[old_index] = generated_index
        occupied_dummy_numbers.add(candidate_number)
        candidate_number += 1

    return term.subs (substitutions)


    # dimensionedScalar vUnit("vUnit", dimLength/dimTime, 1);

    # GeometricField<scalar, PatchType, GeoMesh> cSqrByRT 
    #     = magSqr(U - xi_)/(R*T);

    # GeometricField<scalar, PatchType, GeoMesh> cqBy5pRT 
    #     = ((xi_ - U)&q)/(5.0*rho*R*T*R*T);

    # GeometricField<scalar, PatchType, GeoMesh> gEqBGK 
    #     = rho/pow(sqrt(2.0*pi*R*T),D)*exp(-cSqrByRT/2.0)/pow(vUnit, 3-D);

    # gEq = ( 1.0 + (1.0 - Pr)*cqBy5pRT*(cSqrByRT - D - 2.0) )*gEqBGK;
    # hEq = ( (K + 3.0 - D) + (1.0 - Pr)*cqBy5pRT*((cSqrByRT - D)*(K + 3.0 - D) - 2*K) )*gEqBGK*R*T;