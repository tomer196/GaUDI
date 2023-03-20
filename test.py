import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from data.gor2goa import rotation_2d

a = np.array([[1.34, 0], [1.34, 1.34], [0, 1.34], [0, 0]])


plt.scatter(a[:, 0], a[:, 1])
plt.gca().set_aspect("equal", adjustable="box")
plt.show()

a -= a.mean(axis=0)
angle = np.arctan2(a[-1, 1], a[-1, 0])
a = a @ rotation_2d(angle)


plt.scatter(a[:, 0], a[:, 1])
plt.gca().set_aspect("equal", adjustable="box")
plt.show()

print(print(np.array2string(a, separator=",")))
