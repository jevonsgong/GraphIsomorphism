# GraphIsomorphism

The main file contains functions for checking isomorphism. For 
a direct use, simply type in NetworkX graphs or use any existing datasets of G(V,E) in the format of a NetworkX graph.
Then call is_isomorphic(G1,G2) would give the result.

Utility functions are in utils.py. Test.py offers a simple test using generated tuples of graphs(G1,G2,G3), where
G2 is a relabeling of G1 and G3 is guranteed to be non-isomorphic to G1. Run test.py will give the testing accuracy of the algorithm compared with the official implementation in NetworkX.
One could change the parameters in test.py (e.g. amount, n,p in erdos_renyi graph) for further testing.
