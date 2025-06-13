from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
a = Permutation([0, 2, 1, 3])
b = Permutation([2, 0, 3, 1])
c = Permutation([0, 3, 2, 1])
G = PermutationGroup([a, b])
G2 = PermutationGroup()
print(G.generators)
G.schreier_sims()

print(G.base)
print(G.strong_gens)
G._generators.append(c)
base, strong_gens = G.schreier_sims_incremental(gens=G.generators+[c])
print(base)
print(strong_gens)

print(*[0,1,2])

