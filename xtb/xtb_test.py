from xtb.interface import Calculator
from xtb.utils import get_method
import numpy as np
numbers = np.array([8, 1, 1])
positions = np.array([
[ 0.00000000000000, 0.00000000000000,-0.73578586109551],
[ 1.44183152868459, 0.00000000000000, 0.36789293054775],
[-1.44183152868459, 0.00000000000000, 0.36789293054775]])

calc = Calculator(get_method("GFN2-xTB"), numbers, positions)
res = calc.singlepoint()  # energy printed is only the electronic part
print(res.get_energy())
print(res.get_gradient())
print(res.get_charges())