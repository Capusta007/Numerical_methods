import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


a2 = lambda x, t: x
phi = lambda x, t: -3*np.exp(3*t) - 4*x*np.exp(2*x)

gamma0 = lambda t: 1 - np.exp(3*t)  # u(0,t)
gamma1 = lambda t: np.exp(2) - np.exp(3*t)  # u(1,t)

psi = lambda x: -1 + np.exp(2*x) 

# ТОЧНОЕ РЕШЕНИЕ (АНАЛИТИЧЕСКОЕ)
def exact_solution(x, t):
    return np.exp(2*x) - np.exp(3*t)

# ============================================================================
# ПАРАМЕТРЫ ВЫЧИСЛИТЕЛЬНОЙ СЕТКИ
# ============================================================================
x_min, x_max = 0.0, 1.0  # Пространственный интервал
t_min, t_max = 0.0, 0.1  # Временной интервал
h = 0.1                  # Шаг по пространству
tau_explicit = 0.001     # Шаг по времени для явной схемы 
tau_implicit = 0.01      # Шаг по времени для неявной схемы 

# Создание равномерной пространственной сетки
# x_m = m·h, где m = 0, 1, ..., M
x_nodes = np.arange(x_min, x_max + h/2, h)
M = len(x_nodes) - 1  # Количество интервалов по пространству


"""
Явная схема имеет вид:
u_m^{n+1} = u_m^n + s·(u_{m+1}^n - 2u_m^n + u_{m-1}^n) + τ·φ(x_m, t_n)
где s = τ/h² * a²(x_m, t_n)
"""
def explicit_scheme():
    tau = tau_explicit
    # Число временных слоев: N = T/τ + 1
    N = int((t_max - t_min) / tau) + 1
    # Матрица решения: u[n, m] - значение на n-м временном слое в точке x_m
    u_explicit = np.zeros((N, M+1))
    
    # НАЧАЛЬНОЕ УСЛОВИЕ: u(x,0) = ψ(x)
    for m in range(M+1):
        u_explicit[0, m] = psi(x_nodes[m])
    
    # ШАГИ ПО ВРЕМЕНИ
    for n in range(N-1):
        t_n = n * tau  # Текущее время
        
        # ГРАНИЧНЫЕ УСЛОВИЯ НА СЛЕДУЮЩЕМ ВРЕМЕННОМ СЛОЕ
        u_explicit[n+1, 0] = gamma0(t_n + tau)    # Левая граница
        u_explicit[n+1, M] = gamma1(t_n + tau)    # Правая граница
        
        # ВНУТРЕННИЕ УЗЛЫ (m = 1, 2, ..., M-1)
        for m in range(1, M):
            s = tau / h**2 * a2(x_nodes[m], t_n)
            
            # ЯВНАЯ СХЕМА (аналог (3.4) из методички)
            u_explicit[n+1, m] = (u_explicit[n, m] + 
                                 s * (u_explicit[n, m+1] - 2*u_explicit[n, m] + u_explicit[n, m-1]) + 
                                 tau * phi(x_nodes[m], t_n))
    
    # Возвращаем решение на последнем временном слое и параметры расчета
    return u_explicit[-1, :], tau, N-1


"""
Неявная схема имеет вид:
-s·u_{m-1}^{n+1} + (1+2s)·u_m^{n+1} - s·u_{m+1}^{n+1} = u_m^n + τ·φ(x_m, t_{n+1})
где s = τ/h² * a²(x_m, t_{n+1})
"""
def implicit_scheme():
    tau = tau_implicit
    N = int((t_max - t_min) / tau) + 1
    u_implicit = np.zeros((N, M+1))
    
    # НАЧАЛЬНОЕ УСЛОВИЕ
    for m in range(M+1):
        u_implicit[0, m] = psi(x_nodes[m])
    
    # ШАГИ ПО ВРЕМЕНИ
    for n in range(N-1):
        t_n = n * tau
        
        # ГРАНИЧНЫЕ УСЛОВИЯ НА НОВОМ СЛОЕ
        u_implicit[n+1, 0] = gamma0(t_n + tau)
        u_implicit[n+1, M] = gamma1(t_n + tau)
        
        # СОЗДАНИЕ ПОЛНОЙ МАТРИЦЫ КОЭФФИЦИЕНТОВ
        size = M - 1  # количество внутренних точек
        A = np.zeros((size, size))  # полная матрица
        F = np.zeros(size)          # правая часть
        
        for i, m in enumerate(range(1, M)):
            s = tau / h**2 * a2(x_nodes[m], t_n + tau)
            
            # Главная диагональ
            A[i, i] = 1 + 2*s
            
            # Нижняя диагональ
            if i > 0:
                A[i, i-1] = -s
            
            # Верхняя диагональ
            if i < size - 1:
                A[i, i+1] = -s
            
            # Правая часть
            F[i] = u_implicit[n, m] + tau * phi(x_nodes[m], t_n + tau)
        
        # Учет граничных условий
        s_first = tau / h**2 * a2(x_nodes[1], t_n + tau)
        F[0] += s_first * u_implicit[n+1, 0]
        
        s_last = tau / h**2 * a2(x_nodes[M-1], t_n + tau)
        F[-1] += s_last * u_implicit[n+1, M]
        
        # РЕШЕНИЕ СИСТЕМЫ ЛИНЕЙНЫХ УРАВНЕНИЙ
        solution = np.linalg.solve(A, F)
        
        # Записываем решение
        u_implicit[n+1, 1:M] = solution
    
    return u_implicit[-1, :], tau, N-1

# ============================================================================
# ВЫЧИСЛЕНИЕ РЕШЕНИЙ И АНАЛИЗ ПОГРЕШНОСТЕЙ
# ============================================================================


# 1. ВЫЧИСЛЕНИЕ ЯВНОЙ СХЕМОЙ
u_explicit_final, tau_e, steps_e = explicit_scheme()

# 2. ВЫЧИСЛЕНИЕ НЕЯВНОЙ СХЕМОЙ
u_implicit_final, tau_i, steps_i = implicit_scheme()

# 3. ТОЧНОЕ РЕШЕНИЕ В КОНЕЧНЫЙ МОМЕНТ ВРЕМЕНИ
t_final = t_max
u_exact_final = exact_solution(x_nodes, t_final)

# 4. ВЫЧИСЛЕНИЕ ПОГРЕШНОСТЕЙ
error_explicit = np.abs(u_explicit_final - u_exact_final)
error_implicit = np.abs(u_implicit_final - u_exact_final)

# ВЫВОД РЕЗУЛЬТАТОВ В ВИДЕ ТАБЛИЦЫ
print("РЕЗУЛЬТАТЫ В МОМЕНТ ВРЕМЕНИ t = 0.1")

# Подготовка данных для таблицы
table_data = []
for m in range(M+1):
    table_data.append([
        x_nodes[m],                    # Координата x
        u_explicit_final[m],           # Решение по явной схеме
        u_implicit_final[m],           # Решение по неявной схеме
        u_exact_final[m],              # Точное решение
        error_explicit[m],             # Погрешность явной схемы
        error_implicit[m]              # Погрешность неявной схемы
    ])

headers = ["x", "Явная схема", "Неявная схема", "Точное решение", 
           "Погр. явная", "Погр. неявная"]

print(tabulate(table_data, headers=headers, floatfmt=".6f", tablefmt="grid"))


# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Сравнение разностных схем для параболического уравнения', fontsize=16)

# ГРАФИК 1: СРАВНЕНИЕ РЕШЕНИЙ
axes[0].plot(x_nodes, u_explicit_final, 'b-o', label='Явная схема', linewidth=2, markersize=6)
axes[0].plot(x_nodes, u_implicit_final, 'r-s', label='Неявная схема', linewidth=2, markersize=6)
axes[0].plot(x_nodes, u_exact_final, 'g--', label='Точное решение', linewidth=3)
axes[0].set_xlabel('Координата x', fontsize=12)
axes[0].set_ylabel('Температура u(x, 0.1)', fontsize=12)
axes[0].set_title('Распределение температуры в момент t = 0.1', fontsize=14)
axes[0].legend(fontsize=10, loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 1])

# ГРАФИК 2: ПОГРЕШНОСТИ
axes[1].plot(x_nodes, error_explicit, 'b-o', label='Явная схема', linewidth=2, markersize=6)
axes[1].plot(x_nodes, error_implicit, 'r-s', label='Неявная схема', linewidth=2, markersize=6)
axes[1].set_xlabel('Координата x', fontsize=12)
axes[1].set_ylabel('Абсолютная погрешность', fontsize=12)
axes[1].set_title('Погрешности численных решений', fontsize=14)
axes[1].legend(fontsize=10, loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 1])

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)  # Место для подписи
plt.show()