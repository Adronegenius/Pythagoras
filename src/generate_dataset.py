import numpy as np
import pandas as pd

a = np.random.uniform(1, 100, 1000)
b = np.random.uniform(1, 100, 1000)
c = np.sqrt(a**2 + b**2)

df = pd.DataFrame({'a': a, 'b': b, 'c': c})
df.to_csv('../data/triangles.csv', index=False)