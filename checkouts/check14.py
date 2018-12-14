import numpy as np

# pth1 = '/home/matvei/qscheme/before_bs.txt'
pth1 = '/home/matvei/qscheme/before_phase.txt'
with open(pth1, 'r') as f:
    s_py = f.read()

# pth2 = '/home/matvei/qscheme/cpp_optimized/results/before_bs.txt'
pth2 = '/home/matvei/qscheme/cpp_optimized/results/before_phase.txt'
with open(pth2, 'r') as f:
    s_cpp = f.read()

py_arr = s_py.split(';')[:-1]
cpp_arr = s_cpp.split(';')[:-1]


cpp_nums = np.array([float(s.split(',')[0][1:]) + 1j*float(s.split(',')[1][:-1]) for s in cpp_arr])
py_nums = np.array([float(s.split(',')[0][1:]) + 1j*float(s.split(',')[1][:-1]) for s in py_arr])

diff = cpp_nums - py_nums
np.max(np.abs(diff))

np.sum(np.abs(diff))


np.sum(np.imag(py_nums))