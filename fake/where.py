from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
result = np.where(arr > 2, 0, 1)

print(result)
axfreq = plt.axes([0.1, 0.1, 0.8, 0.05])
slider = Slider(
    ax=axfreq,
    label="Offset1",
    valmin=-5,
    valmax=5,
    valinit=0,
)
# plt.subplots_adjust(left=0.1, bottom=0.25)


axfreq = plt.axes([0.1, 0.2, 0.8, 0.1])
slider = Slider(
    ax=axfreq,
    label="Offset2",
    valmin=-5,
    valmax=5,
    valinit=0,
)
# plt.subplots_adjust(left=0.1, bottom=0.5)
plt.show()

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("axis=0,", arr.sum(axis=0))
print("axis=1,", arr.sum(axis=1))

# 如果指定后续参数，则修改原来的矩阵
result = np.where(arr % 2 == 0, 'numpywhere', 'geek-docs')
print(result)
'''
[['geek-docs' 'numpywhere' 'geek-docs']
 ['numpywhere' 'geek-docs' 'numpywhere']
 ['geek-docs' 'numpywhere' 'geek-docs']]
'''

result = np.where(arr % 2 == 0)
# 如果不指定后续的参数，则返回符合条件的x和y位置
print(result)
'''
(array([0, 1, 1, 2], dtype=int64), array([1, 0, 2, 1], dtype=int64))
'''