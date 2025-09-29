import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sympy import symbols, Function, dsolve, Eq, diff
from scipy.integrate import quad


h = 0.01
a = 0
b = 1

alpha_0 = 4
beta_0 = -2
gamma_0 = -5
alpha_1 = 1
beta_1 = 1
gamma_1 = 6

def p(x): return x + 2
def q(x): return 2
def f(x): return 1



############################
# TASK1: Метод вариации произвольных постоянных
############################
print("\n############ TASK1: Метод вариации произвольных постоянных ############")

def euler_system(F, x0, Y0, h, X):
    x_vals = [x0]
    Y_vals = [Y0]
    x = x0
    Y = np.array(Y0, dtype=float)
    N = int((X - x0) / h)
    for _ in range(N):
        Y = Y + h * np.array(F(x, Y))
        x = x + h
        x_vals.append(x)
        Y_vals.append(Y.copy())
    return np.array(x_vals), np.array(Y_vals)

# Система для Z
def F0(x, Y):
    T, S = Y
    return [S, -p(x)*S + q(x)*T + f(x)]
# Z(0)=0, Z'(0)=0
x0_vals, Y0_vals = euler_system(F0, a, [0, 0], h, b)

# Система для Z1
def F1(x, Y):
    T, S = Y
    return [S, -p(x)*S + q(x)*T]
# Z1(0)=0, Z1'(0)=1
x1_vals, Y1_vals = euler_system(F1, a, [0, 1], h, b)

# Система для Z2
def F2(x, Y):
    T, S = Y
    return [S, -p(x)*S + q(x)*T]
# Z2(0)=1, Z2'(0)=0
x2_vals, Y2_vals = euler_system(F2, a, [1, 0], h, b)

# В конце интервала
Z0b, Z0b_der = Y0_vals[-1]
Z1b, Z1b_der = Y1_vals[-1]
Z2b, Z2b_der = Y2_vals[-1]

# Составляем систему для C1, C2
A1 = alpha_0*0 + beta_0*1   # при Z1
A2 = alpha_0*1 + beta_0*0   # при Z2
RHS1 = gamma_0 - (alpha_0*0 + beta_0*0)

B1 = alpha_1*Z1b + beta_1*Z1b_der
B2 = alpha_1*Z2b + beta_1*Z2b_der
RHS2 = gamma_1 - (alpha_1*Z0b + beta_1*Z0b_der)

M = np.array([[A1, A2],[B1, B2]], dtype=float)
rhs = np.array([RHS1, RHS2], dtype=float)
C1, C2 = np.linalg.solve(M, rhs)

print(f"C1 = {C1:.6f}, C2 = {C2:.6f}")

# Решение
y_vals = C1*Y1_vals[:,0] + C2*Y2_vals[:,0] + Y0_vals[:,0]

# Таблица значений
data = []
for k in range(0, 11):
    idx = int(k*0.1/h)
    data.append([k, x0_vals[idx], y_vals[idx]])
print(tabulate(data, headers=['k','x(k)','y(xk)'], tablefmt='grid', floatfmt=".6f"))

plt.figure(figsize=(10,5))
plt.plot(x0_vals, y_vals, 'b-', label="Метод вариации постоянных")
plt.grid(True)
plt.legend()
plt.title("TASK1: Вариация постоянных")


############################
# TASK2: Метод Галеркина (n=4)
############################
print("\n############ TASK2: Метод Галеркина (n=4) ############")

# Базисные функции φ_k(x)
phi = [
    lambda x: (1 - x),          # φ0
    lambda x: x*(1 - x),        # φ1
    lambda x: x**2*(1 - x),     # φ2
    lambda x: x**3*(1 - x),     # φ3
    lambda x: x**4*(1 - x)      # φ4
]

# Их производные
dphi = [
    lambda x: -1,                  # dφ0/dx
    lambda x: 1 - 2*x,             # dφ1/dx
    lambda x: 2*x - 3*x**2,        # dφ2/dx
    lambda x: 3*x**2 - 4*x**3,     # dφ3/dx
    lambda x: 4*x**3 - 5*x**4      # dφ4/dx
]

n = 4
A = np.zeros((n, n))
d = np.zeros(n)

for i in range(n):
    for j in range(n):
        integrand = lambda x: (p(x)*phi[j](x) - dphi[j](x)) * phi[i](x)
        A[i, j] = quad(integrand, a, b)[0]
    d[i] = quad(lambda x: f(x)*phi[i](x), a, b)[0]

a_coeffs = np.linalg.solve(A, d)
print("Коэффициенты метода Галеркина:", a_coeffs)

x_g = np.linspace(0, 1, 200)
y_g = np.zeros_like(x_g)
for k in range(n):
    y_g += a_coeffs[k] * np.array([phi[k](xx) for xx in x_g])

plt.plot(x_g, y_g, 'r--', label="Метод Галеркина n=4")
plt.legend()


############################
# TASK3: Точное решение через Sympy
############################
# x = symbols('x')
# y = Function('y')
# ode = Eq(diff(y(x),x,2) + p(x)*y(x) - q(x)*diff(y(x),x), f(x))
# sol = dsolve(ode)
# print("\nОбщее решение ODE:", sol)

plt.show()

