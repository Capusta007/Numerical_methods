import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# --- Параметры ---
h = 0.1
x0 = 0
X = 1
x = np.arange(x0, X, h)
m = 1
a = 0.3*m

a_0 = 1
b_0 = 0
a_1 = 1
b_1 = 1 + a

gamma_0 = -1/a
gamma_1 = a*(a+2)

def p(x): return pow(x+a,2)
def q(x): return 2 / pow(x+a,2)
def f(x): return -2*a*x / pow(x+a,2) + a*pow(x+a,2) + 1

# --- Метод Рунге–Кутты 2 порядка для системы ---
def runge_kutta_system(ode_func, x0, y0, h, X):
    x_vals = [x0]
    y_vals = [y0.copy()]
    
    x_current = x0
    y_current = y0.copy()
    
    while x_current < X:
        k1 = h * ode_func(x_current, y_current)
        k2 = h * ode_func(x_current + h, y_current + k1)
        
        y_current += (k1 + k2) / 2
        x_current += h
        
        x_vals.append(x_current)
        y_vals.append(y_current.copy())
    
    return np.array(x_vals), np.array(y_vals)

# --- Система ОДУ ---
def ode_system(x, y):
    # y = [u, v] = [y(x), y'(x)]
    u = y[0]  # это y(x)
    v = y[1]  # это y'(x)
    
    du_dx = v  # u' = v (первое уравнение)
    dv_dx = -p(x)*v + q(x)*u + f(x)  # v' = -p(x)v + q(x)u + f(x) (второе уравнение)
    
    return np.array([du_dx, dv_dx])

# --- Функция невязки для метода пристрелки ---
def shooting_residual(dy0_guess):
    y0 = np.array([gamma_0, dy0_guess])  # y(0) = -1/a, y'(0) = dy0_guess
    x_vals, y_vals = runge_kutta_system(ode_system, x0, y0, h, X)
    residual = a_1 * y_vals[-1, 0] + b_1 * y_vals[-1, 1] - gamma_1
    return residual

# --- Метод пристрелки ---
def shooting_method():
    t0, t1 = 0.0, 1.0
    F0, F1 = shooting_residual(t0), shooting_residual(t1)
    
    t2 = t1 - F1 * (t1 - t0) / (F1 - F0)
    
    y0 = np.array([gamma_0, t2])
    return runge_kutta_system(ode_system, x0, y0, h, X)

# --- Решение с помощью scipy ---
def solve_with_scipy():
    def ode(x, y):
        return np.vstack([y[1], 
                         -p(x)*y[1] + q(x)*y[0] + f(x)])
    
    def bc(ya, yb):
        return np.array([ya[0] - gamma_0,  # y(0) = -1/a
                        a_1*yb[0] + b_1*yb[1] - gamma_1])
    
    y_guess = np.zeros((2, x.size))
    solution = solve_bvp(ode, bc, x, y_guess, tol=1e-6)
    
    return solution.x, solution.y[0]
    

# --- Получение решений ---
x_shooting, y_shooting = shooting_method()
x_scipy, y_scipy = solve_with_scipy()

# --- Построение графика ---
plt.figure(figsize=(10, 6))
plt.plot(x_shooting, y_shooting[:, 0], 'b-', label='Метод пристрелки')
plt.plot(x_scipy, y_scipy, 'r--', label='Scipy solve_bvp')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title(f'Решение граничной задачи 22 (a={a})')
plt.legend()
plt.grid(True)
plt.show()