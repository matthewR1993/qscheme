import numpy as np
import qutip as qt
from math import sqrt
import matplotlib.pyplot as plt


t1 = sqrt(0.5)
r1 = sqrt(0.5)

t2 = sqrt(0.5)
r2 = sqrt(0.5)

t3 = sqrt(0.5)
r3 = sqrt(0.5)

t4 = sqrt(0.5)
r4 = sqrt(0.5)


# 1st detector was hit
# first coef.
def click1_prob1(t1, t4):
    r1 = 1 - t1
    r4 = 1 - t4
    return pow(abs((t1*t1 - r1*r1)*(t2*t3 + 1j*t3*t2)*t4 - t1*r1*(t3*t3 + 1j*2*t3*r3)*r4), 2)

# dependence on t1
t1_list = np.linspace(0.5, 1, 100)
p1_list = click1_prob1(t1_list, t4)

plt.plot(t1_list, p1_list)
plt.show()

# dependence on t4
t4_list = np.linspace(0.5, 1, 100)
p1_list = click1_prob1(t1, t4_list)

plt.plot(t4_list, p1_list)
plt.show()


# 3rd detector was hit
# first coef
def click3_prob1(r1, t4):
    r1 = 1 - t1
    r4 = 1 - t4
    return pow(abs((t1*t1 - r1*r1)*(t2*t3 + 1j*t2*t3)*t4 - t1*r1*(t2*t2 + 1j*2*t2*r2)*r4), 2)

# dependence on t1
t1_list = np.linspace(0.5, 1, 100)
p1_list = click3_prob1(t1_list, t4)

plt.plot(t1_list, p1_list)
plt.show()

# dependence on t4
t4_list = np.linspace(0.5, 1, 100)
p1_list = click3_prob1(t1, t4_list)

plt.plot(t4_list, p1_list)
plt.show()
