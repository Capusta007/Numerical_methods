import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.integrate import solve_ivp

# Параметры задачи (вариант 10)
a = 1.0
b = 2.5
c = 3.55
f = -1.4
q = 1.91

# Параметры сетки
h = 0.01  # шаг по x
tau = 0.005  # шаг по t
x0, xN = 0.0, 1.0  # границы по x
t0, tN = 0.0, 0.5  # границы по t

# Создание сетки
x = np.arange(x0, xN + h, h)
t = np.arange(t0, tN + tau, tau)
M = len(x) - 1  # количество интервалов по x
N = len(t) - 1  # количество интервалов по t

# Инициализация сеточной функции
u = np.zeros((M + 1, N + 1))

# Граничные условия
for n in range(N + 1):
    u[0, n] = f + np.sin(np.pi * t[n])  # u(0,t)
    u[M, n] = f + t[n] ** 3  # u(1,t)

# Начальные условия
for m in range(M + 1):
    u[m, 0] = x[m] ** 2 - x[m] + f  # u(x,0)

# Вычисление u на первом временном слое
for m in range(1, M):
    u[m, 1] = u[m, 0] + tau * (f**2 + np.exp(q * x[m]))  # du/dt(x,0) = f^2 + e^{qx}

# Коэффициент s = (τ^2 / h^2) * a^2(x,t)
s = tau**2 / h**2

# Функция правой части уравнения (at+b)/(1+c^2*x^2)
def phi(x_val, t_val):
    return (a * t_val + b) / (1 + c**2 * x_val**2)

# Явная схема
for n in range(1, N):
    for m in range(1, M):
        u[m, n+1] = (s * (u[m+1, n] + u[m-1, n]) +
                     2 * (1 - s) * u[m, n] -
                     u[m, n-1] +
                     tau**2 * phi(x[m], t[n]))

# Получение эталонного решения с помощью scipy (решаем как систему ОДУ)
def wave_equation_system(t_span, y, x_grid):
    M_inner = len(x_grid) - 2  # внутренние точки
    u_vals = y[:M_inner+2]  # +2 для границ
    v_vals = y[M_inner+2:]
    
    dudt = v_vals.copy()
    dvdt = np.zeros_like(v_vals)
    
    # Вторая производная по x (центральная разность)
    for i in range(1, M_inner+1):
        d2u_dx2 = (u_vals[i+1] - 2*u_vals[i] + u_vals[i-1]) / h**2
        dvdt[i] = d2u_dx2 + phi(x_grid[i], t_span)
    
    return np.concatenate([dudt, dvdt])

# Подготовка данных для scipy
x_inner = x[1:-1]  # внутренние точки
y0 = np.zeros(2 * (len(x_inner) + 2))

# Начальные условия
y0[:len(x_inner)+2] = x**2 - x + f  # u(x,0)
y0[len(x_inner)+2:] = f**2 + np.exp(q * x)  # du/dt(x,0)

# Добавляем граничные условия в начальный вектор
y0[0] = f + np.sin(np.pi * t0)  # u(0,0)
y0[len(x_inner)+1] = f + t0**3  # u(1,0)

# Решаем систему ОДУ
sol = solve_ivp(
    lambda t, y: wave_equation_system(t, y, x),
    [t0, tN],
    y0,
    t_eval=t,
    method='RK45',
    rtol=1e-8,
    atol=1e-10
)

# Извлекаем эталонное решение
u_ref = sol.y[:M+1, :]

# Вычисление абсолютной погрешности
error = np.abs(u - u_ref)

n_temp = N // 5
# Создание таблицы для вывода
table_data = []
for m in range(0, M+1, M//10):  # каждую 10-ю точку по x
    for n in range(0, N+1, N//5):  # каждую 5-ю точку по t
        table_data.append([
            x[m], t[n_temp], 
            u[m, n_temp], u_ref[m, n_temp], 
            error[m, n_temp]
        ])

headers = ["x", "t", "Численное u", "Эталонное u", "Абс. погрешность"]
print("\n" + tabulate(table_data[:15], headers=headers, tablefmt="grid", floatfmt=".6f"))

# График 1: Сравнение численного и эталонного решений
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, u[:, n_temp], 'b-', linewidth=2, label='Численное')
plt.plot(x, u_ref[:, n_temp], 'r--', linewidth=2, label='Эталонное')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title(f'Решение при t = {t[n_temp]:.3f}')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, error[:, n_temp], 'g-', linewidth=2)
plt.xlabel('x')
plt.ylabel('Абсолютная погрешность')
plt.title(f'Погрешность численного решения при t = {t[n_temp]:.3f}')
plt.grid(True)

plt.tight_layout()
plt.show()