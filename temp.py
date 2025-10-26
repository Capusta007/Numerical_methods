import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# --- Параметры ---
h = 0.1
x0 = 0
X = 1
x = np.arange(x0, X, h)
m = 1
a = 1 + 0.1*m

a_0 = 1
b_0 = 0
a_1 = a
b_1 = 1

gamma_0 = 1
gamma_1 = -2*a*np.sin(a)

def p(x): return a
def q(x): return 0
def f(x): return 2*pow(a,2)*np.sin(a*x)

def solve_scipy():
    # --- Решение через solve_bvp ---
    def fun_bvp(x, y):
        return np.vstack([y[1], -p(x)*y[1] + q(x)*y[0] + f(x)])

    def bc_bvp(ya, yb):
        return np.array([a_0*ya[0] + b_0*ya[1] - gamma_0, 
                        a_1*yb[0] + b_1*yb[1] - gamma_1])

    x_bvp = np.linspace(x0, X, 100)
    y0_bvp = np.zeros((2, x_bvp.size))
    return solve_bvp(fun_bvp, bc_bvp, x_bvp, y0_bvp)

sol_bvp = solve_scipy()

# --- Построение графика ---
plt.figure(figsize=(10, 6))
plt.plot(sol_bvp.x, sol_bvp.y[0], 'b-', linewidth=2, label='solve_bvp')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title(f'Решение ДУ: y\'\' + {a}y\' = 2{a}²·sin({a}x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()