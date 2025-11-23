import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

m = 1
a = 0.5*m
h = 0.1
N = int(1 / h)
x = np.linspace(0, 1, N + 1)  

# -----------------------------------------------------------
# Построение трёхдиагональной системы (метод конечных разностей)
#
# Центральные разности:
#   y''(x_i) ≈ (y_{i+1} - 2 y_i + y_{i-1}) / h^2
#   y'(x_i)  ≈ (y_{i+1} - y_{i-1}) / (2h)
#
# Подстановка в уравнение y'' + p(x_i) y' = f(x_i) даёт:
#  A_i y_{i-1} + B_i y_i + C_i y_{i+1} = F_i
#
# где p(x) = 1.5 a / (a x + 1), f(x) = 2 a / sqrt(a x + 1)
# A_i = 1/h^2 - p_i/(2h)
# B_i = -2/h^2
# C_i = 1/h^2 + p_i/(2h)
# -----------------------------------------------------------

A = np.zeros(N + 1)   
F = np.zeros(N + 1)   
B = np.zeros(N + 1)   
C = np.zeros(N + 1)   

# Заполнение коэффициентов в внутренних узлах i=1..N-1
for i in range(1, N):
    p_i = 1.5 * a / (a * x[i] + 1)
    f_i = 2 * a / np.sqrt(a * x[i] + 1)

    A[i] = 1/h**2 - p_i/(2*h)
    B[i] = -2/h**2
    C[i] = 1/h**2 + p_i/(2*h)
    F[i] = f_i

# -----------------------------------------------------------
# Граничные условия - левая и правая
#
# Левое:  3ay(0) - y'(0) = 1
#        используем приближение y'(0) ≈ (y1 - y_{-1})/(2h),
#        выразим y_{-1} через y0 и y1 и исключим его,
#        в результате получаем уравнение вида:
#          (3a + 1/h) y_0  - (1/h) y_1 = 1
#
# Правое: y'(1) = sqrt(a+1)
#        приближение: (y_{N+1} - y_{N-1})/(2h) = sqrt(a+1)
#        => y_{N+1} = y_{N-1} + 2h * sqrt(a+1)
#        подставляем и получаем уравнение, сведённое к y_{N-1} и y_N:
#        можно записать в форме A[N] y_{N-1} + B[N] y_N = F[N]

# Левая граница (строка 0)
B[0] = 3*a + 1/h   # коэффициент при y_0
C[0] = -1/h        # коэффициент при y_1
A[0] = 0
F[0] = 1

# Правая граница (строка N)
A[N] = -1/(2*h)
B[N] = 1/(2*h)
C[N] = 0
F[N] = np.sqrt(a + 1)

# Решение системы методом прогонки
# Система: A[i] y_{i-1} + B[i] y_i + C[i] y_{i+1} = F[i], i=0..N

alpha = np.zeros(N + 1)
beta = np.zeros(N + 1)

# Прямой ход
# Для i = 0:
alpha[0] = -C[0] / B[0]
beta[0] = F[0] / B[0]

# Для i = 1..N
for i in range(1, N + 1):
    denom = B[i] + A[i] * alpha[i - 1]
    if i < N:
        alpha[i] = -C[i] / denom
    beta[i] = (F[i] - A[i] * beta[i - 1]) / denom

# Обратный ход
y_fd = np.zeros(N + 1)
y_fd[N] = beta[N]
for i in range(N - 1, -1, -1):
    y_fd[i] = alpha[i] * y_fd[i + 1] + beta[i]

# Точное решение

def odefun(xv, z):
    p = 1.5 * a / (a * xv + 1)
    f = 2 * a / np.sqrt(a * xv + 1)
    dy1 = z[1]
    dy2 = f - p * z[1]   # из y'' + p y' = f  =>  y'' = f - p y'
    return [dy1, dy2]

target_deriv = np.sqrt(a + 1.0)

def shoot_residual(s):
    y0 = (1 + s) / (3.0 * a)  
    z0 = [y0, s]
    # интегрируем достаточно точно
    sol = solve_ivp(odefun, (0.0, 1.0), z0, t_eval=[1.0], atol=1e-10, rtol=1e-10)
    yprime_at_1 = sol.y[1, -1]
    return yprime_at_1 - target_deriv

s_samples = np.linspace(-50, 50, 501)
vals = [shoot_residual(sv) for sv in s_samples]

bracket = None
for k in range(len(s_samples) - 1):
    if vals[k] == 0:
        bracket = (s_samples[k], s_samples[k])
        break
    if vals[k] * vals[k + 1] < 0:
        bracket = (s_samples[k], s_samples[k + 1])
        break

if bracket is None:
    bracket = (-1.0, 1.0)

res = root_scalar(shoot_residual, bracket=bracket, method='bisect', xtol=1e-12, rtol=1e-12, maxiter=200)
if not res.converged:
    raise RuntimeError("Не удалось найти начальное значение s методом стрельбы.")

s_star = res.root
y0_star = (1 + s_star) / (3.0 * a)
sol_dense = solve_ivp(odefun, (0.0, 1.0), [y0_star, s_star], t_eval=np.linspace(0.0, 1.0, 2000),
                      atol=1e-12, rtol=1e-12)
x_dense = sol_dense.t
y_dense = sol_dense.y[0, :]

y_ref_nodes = np.interp(x, x_dense, y_dense) 


# Таблица сравнения и графики

errors = np.abs(y_ref_nodes - y_fd)

# Таблица: x, численное (сеточное), эталонное (solve_ivp), ошибка
table = []
for i in range(N + 1):
    table.append([f"{x[i]:.1f}", f"{y_fd[i]:.6f}", f"{y_ref_nodes[i]:.6f}", f"{errors[i]:.2e}"])

print("\nСравнение сеточного и эталонного решения (метод прогонки vs solve_ivp):")
print(tabulate(table, headers=["x", "y сеточное", "y эталон", "абс.ошибка"], tablefmt="github"))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True) # type: ignore

ax1.plot(x_dense, y_dense, label="Эталон (solve_ivp, плотная сетка)", linewidth=2)
ax1.plot(x, y_fd, 'o-', label="Сеточное (метод прогонки, h=0.1)", markersize=6)
ax1.set_ylabel("y")
ax1.set_title("Сравнение: эталонное решение vs сеточное")
ax1.grid(True)
ax1.legend()

ax2.plot(x, errors, 'r.--', linewidth=1.5, markersize=6)
ax2.set_xlabel("x")
ax2.set_ylabel("абс.ошибка")
ax2.set_title("Погрешность |y_ref - y_fd| в узлах сетки")
ax2.grid(True)

plt.tight_layout()
plt.show()
