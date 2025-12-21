import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from tabulate import tabulate

# Параметры задачи
c = 1.0  # скорость распространения волны
L = 10.0  # длина расчетной области (-L, L)
T = 2.0   # конечное время

# Параметры сетки
h = 0.1   # шаг по пространству
tau = 0.05  # шаг по времени

# Создание сетки
x = np.arange(-L, L + h, h)
t = np.arange(0, T + tau, tau)

M = len(x) - 1  # количество интервалов по x
N = len(t) - 1  # количество интервалов по t

# Инициализация сеточной функции
u_num = np.zeros((M + 1, N + 1))  # численное решение

# Начальные условия
def f(x_val):
    """Начальная форма струны"""
    return 0.0

def g(x_val):
    """Начальная скорость точек струны"""
    if -1.0 < x_val < 1.0:
        return 1.0
    return 0.0

# Применение начальных условий
for m in range(M + 1):
    u_num[m, 0] = f(x[m])

# Используем разностную схему для первого временного слоя
for m in range(1, M):
    u_num[m, 1] = u_num[m, 0] + tau * g(x[m])

# Граничные условия для бесконечной струны (аппроксимация)
# Используем условие прозрачности (свободные границы)
for n in range(N + 1):
    u_num[0, n] = u_num[1, n]  # слева
    u_num[M, n] = u_num[M-1, n]  # справа

# Коэффициент для разностной схемы
s = (c * tau / h) ** 2

# Основной цикл по времени
for n in range(1, N):
    for m in range(1, M):
        u_num[m, n+1] = (s * (u_num[m+1, n] + u_num[m-1, n]) +
                         2 * (1 - s) * u_num[m, n] - 
                         u_num[m, n-1])

    u_num[0, n+1] = u_num[1, n+1]   
    u_num[M, n+1] = u_num[M-1, n+1] 

# Функция Даламбера для аналитического решения
def u_exact(x_val, t_val):
    """Аналитическое решение по формуле Даламбера"""
    term1 = 0.5 * (f(x_val - c * t_val) + f(x_val + c * t_val))
    
    # Вычисление интеграла аналитически
    lower = x_val - c * t_val
    upper = x_val + c * t_val
    
    # Определяем пересечение интервала (-1, 1) с (lower, upper)
    a = max(lower, -1.0)
    b = min(upper, 1.0)
    
    if b > a:
        integral_value = b - a
    else:
        integral_value = 0.0
    
    term2 = (1.0 / (2.0 * c)) * integral_value
    
    return term1 + term2

# Создание массива аналитического решения
u_analytical = np.zeros((M + 1, N + 1))
for m in range(M + 1):
    for n in range(N + 1):
        u_analytical[m, n] = u_exact(x[m], t[n])

# Вычисление абсолютной погрешности
error = np.abs(u_num - u_analytical)

# Выбор временных срезов для визуализации
time_indices = [N]
time_labels = [f"t = {t[idx]:.2f}" for idx in time_indices]

# График 1: Сравнение численного и аналитического решений

plt.subplot(1, 2, 1)
plt.plot(x, u_num[:, N], 'b-', linewidth=2, label='Численное')
plt.plot(x, u_analytical[:, N], 'r--', linewidth=2, label='Аналитическое')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim([-0.1, 1.1])

plt.suptitle('Сравнение численного и аналитического решений', fontsize=16)

plt.subplot(1, 2, 2)
plt.plot(x, error[:, N], 'g-', linewidth=2)
plt.xlabel('x')
plt.ylabel('Абсолютная погрешность')
plt.grid(True, alpha=0.3)
plt.ylim([0, np.max(error) * 1.1])

plt.suptitle('Абсолютная погрешность численного решения', fontsize=16)
plt.tight_layout()
plt.show()

# Таблица значений в некоторых точках
print("\n" + "="*80)
print("ТАБЛИЦА ЗНАЧЕНИЙ В ВЫБРАННЫХ ТОЧКАХ")
print("="*80)

# Выбираем точки для отображения в таблице
x_indices = range(M//3,3*M//3)  # точки по x
t_indices = [N]     # моменты времени (кроме t=0)


table_data = []
headers = ["x", "t", "Численное", "Аналитическое", "Погрешность"]

for t_idx in t_indices:
    for x_idx in x_indices:
        x_val = x[x_idx]
        t_val = t[t_idx]
        num_val = u_num[x_idx, t_idx]
        ana_val = u_analytical[x_idx, t_idx]
        err_val = error[x_idx, t_idx]
        
        table_data.append([
            f"{x_val:.2f}",
            f"{t_val:.2f}",
            f"{num_val:.6f}",
            f"{ana_val:.6f}",
            f"{err_val:.6e}"
        ])

print(tabulate(table_data, headers=headers, tablefmt="grid"))