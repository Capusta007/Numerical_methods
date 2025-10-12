import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp

print("########################")
print("### ЛАБОРАТОРНАЯ №8 ###")
print("### Вариант №10     ###")
print("########################\n")

a, b = 0, 1
h = 0.01

def p(x): return x + 2
def q(x): return 3
def f(x): return 1

alpha0, beta0, gamma0 = 4, -2, -5
alpha1, beta1, gamma1 = 1, 1, 6






#############ТОЧНОЕ РЕШЕНИЕ#############
def f_sys(x, Y):
    y = Y[0]
    z = Y[1]
    return np.vstack((z, -p(x)*z + q(x)*y + f(x)))

# Граничные условия
def bc(Y_a, Y_b):
    y_a, z_a = Y_a
    y_b, z_b = Y_b
    return np.array([
        alpha0*y_a + beta0*z_a - gamma0,
        alpha1*y_b + beta1*z_b - gamma1
    ])
x_init = np.linspace(a, b, 50)
y_init = np.zeros((2, x_init.size))
sol = solve_bvp(f_sys, bc, x_init, y_init)

# Получаем точное решение
x_dense = np.linspace(a, b, 500)
y_vals = sol.sol(x_dense)[0]  









#############МЕТОД ВАРИАЦИИ ПРОИЗВОЛЬНЫХ ПОСТОЯННЫХ#############
print("#############TASK1#############")

def euler_system(f1, f2, x0, y0, z0, h, X):
    x = [x0]
    y = [y0]
    z = [z0]
    N = int((X - x0) / h)
    for _ in range(N):
        y0 += h * f1(x0, y0, z0)
        z0 += h * f2(x0, y0, z0)
        x0 += h
        x.append(x0)
        y.append(y0)
        z.append(z0)
    return np.array(x), np.array(y), np.array(z)

# Надо решить уравненение y'' + p(x)y' -q(x)y = f(x)
# Делаем замену y' = z
# И получается система: (их 3 будет, тут только 1я расписана)
# y' = z
# z' = -p(x)z + q(x)y + f(x)
def fZ(x, y, z): return -p(x) * z + q(x) * y + f(x)
def fZ1(x, y, z): return -p(x) * z + q(x) * y
def fZ2(x, y, z): return -p(x) * z + q(x) * y


# Решаем 3 задачи Коши
x, y, z = euler_system(lambda x, y, z: z, fZ, a, 0, 0, h, b)
x, y1, z1 = euler_system(lambda x, y, z: z, fZ1, a, 0, 1, h, b)
x, y2, z2 = euler_system(lambda x, y, z: z, fZ2, a, 1, 0, h, b)

# Система для C1, C2
# Z1 = y1, Z1' = z1
A11 = alpha0*y1[0] + beta0*z1[0]
A12 = alpha0*y2[0] + beta0*z2[0]
B1  = gamma0 - (alpha0*y[0] + beta0*z[0])

A21 = alpha1*y1[-1] + beta1*z1[-1]
A22 = alpha1*y2[-1] + beta1*z2[-1]
B2  = gamma1 - (alpha1*y[-1] + beta1*z[-1])

C1, C2 = np.linalg.solve([[A11, A12], [A21, A22]], [B1, B2])
print(f"C1 = {C1:.5f}, C2 = {C2:.5f}\n")

# Решение методом вариации
Y = y + C1 * y1 + C2 * y2

# График
plt.figure(figsize=(10,6))
plt.plot(x, Y, 'o-', label='Метод вариации постоянных (Эйлер)')
plt.plot(x_dense, y_vals, '-', label='Точное решение (solve_bvp)')
plt.legend()
plt.title("Метод вариации произвольных постоянных (Вариант 10)")
plt.grid(True)
plt.show()













##########################
# МЕТОД ГАЛЕРКИНА (n=4)
##########################
print("\n=== Метод Галеркина (n=4) ===")

phi = [
    lambda x: x**2 - x,
    lambda x: x**3 - x**2,
    lambda x: x**4 - x**3,
    lambda x: x**5 - x**4
]

n = len(phi)
A = np.zeros((n, n))
B = np.zeros(n)
xs = np.linspace(a, b, 200)

def Lphi(func, x):
    dx = 1e-5
    f = func(x)
    df = (func(x + dx) - func(x - dx)) / (2 * dx)
    d2f = (func(x + dx) - 2*func(x) + func(x - dx)) / (dx**2)
    return d2f + p(x)*df - q(x)*f

for i in range(n):
    for j in range(n):
        A[i, j] = np.trapz(Lphi(phi[j], xs) * phi[i](xs), xs)
    B[i] = np.trapz(f(xs) * phi[i](xs), xs)

a_coeff = np.linalg.solve(A, B)
Yg = np.zeros_like(x)
for k in range(n):
    Yg += a_coeff[k] * phi[k](x)

# Добавим линейную комбинацию, чтобы удовлетворить граничным условиям
Yg += (gamma0/alpha0) * (1 - x) + (gamma1/alpha1) * x

plt.figure(figsize=(10,6))
plt.plot(x, Yg, 'o-', label='Метод Галеркина (n=4)')
plt.plot(x_dense, y_vals, '-', label='Точное решение (solve_bvp)')
plt.legend()
plt.title("Метод Галеркина (Вариант 10)")
plt.grid(True)
plt.show()
