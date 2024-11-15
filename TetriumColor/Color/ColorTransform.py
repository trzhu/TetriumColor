"""
Goal: Sample Directions in Color Space. 
1. Given HSV coordinates, find their equivalent in display space
2. Transform from HSV (not a linear transform)-> MaxBasis->Display Space (Given Transform)
3. If outside of range [0, 1], clamp or return some sort of threshold reached error. 
3.5 -- probably want to precompute the bounds on each direction so quest doesn't try to keep testing useless saturations
"""

#TODO: Implement the above, also generate all of the precomputed matrices. We also need to account for S-cone noise somewhere, still not applied.