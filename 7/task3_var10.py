import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tabulate import tabulate

# Параметры задачи
L = 1.0          # длина струны
T = 0.5          # конечное время
h = 0.1          # шаг по x
tau = 0.05       # шаг по t

# Сетки
x_vals = np.arange(0, L + h, h)       
t_vals = np.arange(0, T + tau, tau)   
M = len(x_vals) - 1                    # количество интервалов по x
N = len(t_vals) - 1                    # количество интервалов по t

# Коэффициент из условия устойчивости (a^2 = 1 в нашем уравнении)
a = 1.0
s = (tau**2) / (h**2) * (a**2)

def gamma0(t):
    return 0.4

def gamma1(t):
    return 1.2 * t

def alpha(x):
    return (x + 0.4) * np.cos(np.pi * x / 2)

def beta(x):
    return 0.3 * (x**2 + 1)

# Правая часть (в нашем уравнении = 0)
def phi(x, t):
    return 0.0

# ========== ЧИСЛЕННОЕ РЕШЕНИЕ РАЗНОСТНОЙ СХЕМОЙ ==========

# Инициализация сеточной функции
u_num = np.zeros((M + 1, N + 1))

# Граничные условия
for n in range(N + 1):
    t = t_vals[n]
    u_num[0, n] = gamma0(t)
    u_num[M, n] = gamma1(t)

# Начальные условия
for m in range(M + 1):
    x = x_vals[m]
    u_num[m, 0] = alpha(x)
    u_num[m, 1] = alpha(x) + tau * beta(x)

# Основной цикл по времени (явная схема)
for n in range(1, N):
    for m in range(1, M):
        u_num[m, n + 1] = (s * (u_num[m + 1, n] + u_num[m - 1, n]) +
                            2 * (1 - s) * u_num[m, n] -
                            u_num[m, n - 1] +
                            tau**2 * phi(x_vals[m], t_vals[n]))

# ========== ТОЧНОЕ РЕШЕНИЕ С ПОМОЩЬЮ SCIPY (ЭТАЛОН) ==========
def wave_ode_system(t, y, x_grid):
    u = y[:M+1]
    v = y[M+1:]
    u_xx = np.zeros_like(u)
    
    # Внутренние точки: вторая производная центральной разностью
    dx = x_grid[1] - x_grid[0]
    for i in range(1, M):
        u_xx[i] = (u[i+1] - 2*u[i] + u[i-1]) / dx**2
    
    # Граничные условия фиксированы
    u_xx[0] = 0
    u_xx[M] = 0
    
    dudt = v
    dvdt = u_xx
    
    return np.concatenate([dudt, dvdt])

# Начальные условия для системы
y0 = np.zeros(2*(M+1))
y0[:M+1] = alpha(x_vals)
y0[M+1:] = beta(x_vals)

# Граничные условия как фиксация значений
def apply_bc(t, y):
    y[0] = gamma0(t)
    y[M] = gamma1(t)
    # Производные на границе тоже фиксируем (из граничных условий)
    y[M+1] = 0  # u_t(0,t) = d/dt gamma0(t) = 0
    y[-1] = 1.2  # u_t(1,t) = d/dt gamma1(t) = 1.2
    return y

# Интегрирование по времени
sol = solve_ivp(
    lambda t, y: wave_ode_system(t, y, x_vals),
    [0, T],
    y0,
    t_eval=t_vals,
    method='RK45',
    rtol=1e-9,
    atol=1e-10,
    max_step=tau/10
)

# Извлечение решения
u_exact = sol.y[:M+1, :]

# ========== НАХОДИМ ИНДЕКС ДЛЯ t = 0.1 ==========
t_target = 0.1
n_index = int(round(t_target / tau))
t_actual = t_vals[n_index]
u_num_t01 = u_num[:, n_index]
u_exact_t01 = u_exact[:, n_index]

# Вычисляем погрешность
error_t01 = np.abs(u_num_t01 - u_exact_t01)
max_error = np.max(error_t01)
avg_error = np.mean(error_t01)

# ========== ТАБЛИЦА ЗНАЧЕНИЙ ПРИ t = 0.1 ==========
table_data = []
for m in range(M + 1):
    x = x_vals[m]
    u_n = u_num_t01[m]
    u_e = u_exact_t01[m]
    err = error_t01[m]
    table_data.append([f"{x:.1f}", f"{u_n:.6f}", f"{u_e:.6f}", f"{err:.6e}"])

print(f"\nТаблица значений при t = {t_actual:.3f}:")
print(tabulate(table_data, 
               headers=["x", "Численное u(x,t)", "Эталонное u(x,t)", "Погрешность |Δu|"],
               tablefmt="grid"))

# ========== ГРАФИКИ ==========
plt.figure(figsize=(14, 4))

# 1. График численного и точного решения
plt.subplot(1, 2, 1)
plt.plot(x_vals, u_num_t01, 'bo-', linewidth=2, markersize=6, label='Численное решение')
plt.plot(x_vals, u_exact_t01, 'r--', linewidth=2, label='Эталонное решение')
plt.xlabel('x', fontsize=12)
plt.ylabel('u(x, t)', fontsize=12)
plt.title(f'Решение при t = {t_actual:.3f}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xticks(np.arange(0, 1.1, 0.1))

# 2. График погрешности
plt.subplot(1, 2, 2)
plt.plot(x_vals, error_t01, 'g-s', linewidth=2, markersize=6)
plt.xlabel('x', fontsize=12)
plt.ylabel('|Числ. - Точн.|', fontsize=12)
plt.title(f'Абсолютная погрешность при t = {t_actual:.3f}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.tight_layout()
plt.show()