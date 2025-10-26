import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# --- Параметры ---
h = 0.1
x0 = 0
X = 1
x = np.arange(x0, X, h)
m = 2
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

# --- Метод Рунге–Кутты 2 порядка для одного уравнения ---
def runge_kutta_2nd_order(ode_func, x0, y0, h, X):
    x_vals = [x0]
    y_vals = [y0]
    
    x_current = x0
    y_current = y0
    
    while x_current < X:
        k1 = h * ode_func(x_current, y_current)
        k2 = h * ode_func(x_current + h, y_current + k1)
        
        y_current += (k1 + k2) / 2
        x_current += h
        
        x_vals.append(x_current)
        y_vals.append(y_current)
    
    return np.array(x_vals), np.array(y_vals)

# --- Метод Рунге–Кутты 2 порядка для ОБРАТНОГО направления ---
def runge_kutta_2nd_order_reverse(ode_func, x_end, y_end, h, x_start):
    x_vals = [x_end]
    y_vals = [y_end]
    
    x_current = x_end
    y_current = y_end
    
    while x_current > x_start + 1e-10:
        k1 = -h * ode_func(x_current, y_current)                    # шаг отрицательный!
        k2 = -h * ode_func(x_current - h, y_current + k1)          # x_current - h
        
        y_current += (k1 + k2) / 2
        x_current -= h  # Двигаемся назад
        
        x_vals.append(x_current)
        y_vals.append(y_current)
    
    return np.array(x_vals), np.array(y_vals)

# --- Метод дифференциальной прогонки ---
def differential_sweep():
    # Прямая прогонка (случай a_0 ≠ 0)
    def ode_u1(x, u1):
        return -u1**2 * q(x) + u1 * p(x) + 1
    
    def ode_u2(x, u2):
        u1_val = np.interp(x, u1_x, u1_vals)
        return -u1_val * (u2 * q(x) + f(x))
    
    # Решаем для u1
    u1_0 = -b_0 / a_0
    u1_x, u1_vals = runge_kutta_2nd_order(ode_u1, 0, u1_0, 0.1, 1)
    
    # Решаем для u2
    u2_0 = gamma_0 / a_0
    u2_x, u2_vals = runge_kutta_2nd_order(ode_u2, 0, u2_0, 0.1, 1)
    
    # Обратная прогонка
    u1_b = u1_vals[-1] #u_1(b)
    u2_b = u2_vals[-1] #u_2(b)
    y_1 = (gamma_1 * u1_b + b_1 * u2_b) / (b_1 + a_1 * u1_b)
    
# Решаем обратную задачу Коши
    def ode_y(x, y):
        u1_val = np.interp(x, u1_x, u1_vals)
        u2_val = np.interp(x, u2_x, u2_vals)
        return (y - u2_val) / u1_val

    # Решаем от x=1 до x=0
    x_reverse, y_reverse = runge_kutta_2nd_order_reverse(ode_y, 1.0, y_1, 0.1, 0.0)

    # Переворачиваем чтобы получить от 0 до 1
    x_sweep = x_reverse[::-1]
    y_sweep = y_reverse[::-1]
    
    return x_sweep, y_sweep




# --- Получение решений ---
x_sweep, y_sweep = differential_sweep()
y_sweep[0]=1

# --- Построение графика ---
plt.figure(figsize=(10, 6))
plt.plot(x_sweep, y_sweep, 'b-', linewidth=2, label='Метод дифференциальной прогонки')
plt.plot(sol_bvp.x, sol_bvp.y[0], 'r--', linewidth=2, label='Scipy solve_bvp')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title(f'Решение задачи 32 (a={a})')
plt.legend()
plt.grid(True)
plt.show()