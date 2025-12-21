import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from tabulate import tabulate

# =============================================
# ФУНКЦИИ ДЛЯ РЕШЕНИЯ УРАВНЕНИЯ ПУАССОНА
# =============================================

def solve_poisson_numerical(h=0.2):
    """Численное решение уравнения Пуассона методом конечных разностей."""
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    
    # Создание сетки
    x = np.arange(x_min, x_max + h/2, h)
    y = np.arange(y_min, y_max + h/2, h)
    nx, ny = len(x), len(y)
    
    # Общее количество узлов
    N = nx * ny
    
    # Функция для преобразования (i,j) в глобальный индекс
    def idx(i, j):
        return i * ny + j
    
    # Построение матрицы и правой части
    A = lil_matrix((N, N))
    F = np.zeros(N)
    coeff = 1.0 / (h**2)
    
    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            
            # Проверка граничных условий
            on_boundary = (i == 0 or i == nx-1 or j == 0 or j == ny-1)
            
            if on_boundary:
                # Граничные условия Дирихле
                A[k, k] = 1.0
                F[k] = 2*(x[i]**2 + y[j])  # u(x,y) = 2(x^2 + y)
            else:
                # Внутренние узлы - уравнение Пуассона
                A[k, k] = -4.0 * coeff
                
                # Соседние узлы
                if i > 0:
                    A[k, idx(i-1, j)] = coeff
                if i < nx-1:
                    A[k, idx(i+1, j)] = coeff
                if j > 0:
                    A[k, idx(i, j-1)] = coeff
                if j < ny-1:
                    A[k, idx(i, j+1)] = coeff
                
                # Правая часть
                F[k] = 4.0
    
    # Решение системы
    A_csr = A.tocsr()
    u_numerical = spsolve(A_csr, F)
    u_num_2d = u_numerical.reshape(nx, ny)
    
    return x, y, u_num_2d, nx, ny, N

def solve_poisson_exact(x, y):
    X, Y = np.meshgrid(x, y, indexing='ij')
    u_exact = 2*X**2 + Y**2 + 2*Y - Y**2
    return u_exact

# Параметры
h = 0.05  # шаг сетки для численного решения

# 1. Численное решение
print("\n1. ВЫЧИСЛЕНИЕ ЧИСЛЕННОГО РЕШЕНИЯ...")
x, y, u_num_2d, nx, ny, N = solve_poisson_numerical(h)

# 2. Аналитическое решение
print("\n2. ВЫЧИСЛЕНИЕ АНАЛИТИЧЕСКОГО РЕШЕНИЯ...")
u_exact_2d = solve_poisson_exact(x, y)

# 3. Вычисление погрешности
error_2d = np.abs(u_num_2d - u_exact_2d)
max_error = np.max(error_2d)
mean_error = np.mean(error_2d)

# 4. Вывод таблицы значений
print("\n4. ТАБЛИЦА ЗНАЧЕНИЙ (первые 5×5 точек):")
print("-" * 70)

table_data = []
for i in range(min(5, nx)):
    for j in range(min(5, ny)):
        x_val = x[i]
        y_val = y[j]
        u_num = u_num_2d[i, j]
        u_exact = u_exact_2d[i, j]
        error_val = error_2d[i, j]
        
        table_data.append([
            f"({x_val:.1f}, {y_val:.1f})",
            f"{u_num:.6f}",
            f"{u_exact:.6f}",
            f"{error_val:.6f}"
        ])

headers = ["Точка (x,y)", "Численное решение", "Аналитическое решение", "Погрешность"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# 5. Построение графиков
print("\n5. ПОСТРОЕНИЕ ГРАФИКОВ...")

fig, axes = plt.subplots(2, figsize=(14, 10))

# График 1: Сравнение при y = 0.5 (центральное сечение)
ax1 = axes[0] # type: ignore
y_fixed = 0.5
j_fixed = np.argmin(np.abs(y - y_fixed))

x_plot = x
u_num_plot = u_num_2d[:, j_fixed]
u_exact_plot = u_exact_2d[:, j_fixed]

ax1.plot(x_plot, u_num_plot, 'bo-', linewidth=2, markersize=8, label='Численное')
ax1.plot(x_plot, u_exact_plot, 'r--', linewidth=2, label='Аналитическое')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel(f'u(x, y={y_fixed})', fontsize=12)
ax1.set_title(f'Сравнение решений при y = {y_fixed}', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# График 2: Погрешность при y = 0.5
ax2 = axes[1] # type: ignore
error_plot = np.abs(u_num_plot - u_exact_plot)

ax2.plot(x_plot, error_plot, 'g-s', linewidth=2, markersize=8)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel(f'Погрешность', fontsize=12)
ax2.set_title(f'Абсолютная погрешность при y = {y_fixed}', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.show()