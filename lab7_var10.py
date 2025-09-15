import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tabulate import tabulate

#################TASK1#################
print("############TASK1############")
h = 0.1
x0 = 1
y0 = 0.5
X = 1.5

def f(x, y) -> float:
    return 0.85 * (x**2) + y/(2*x) - (y**2)/x

def runge_kutta(f, x0:float, y0:float, h:float, X:float):
    x = [x0]
    y = [y0]
    N = int((X - x0) / h)
    for i in range(N):
        K1 = h * f(x0,y0)
        K2 = h * f(x0 + h/2, y0 + K1/2)
        K3 = h * f(x0 + h/2, y0 + K2/2)
        K4 = h * f(x0 + h, y0 + K3)
        dy = (K1 + 2*K2 + 2*K3 + K4)/6
        x0 = x0 + h
        y0 = y0 + dy
        x.append(x0)
        y.append(y0)
    return np.array(x), np.array(y)
    
x, y = runge_kutta(f, x0, y0, h, X)

# Решение через встроенную функцию
y0_ivp = [y0]
sol = solve_ivp(f, [x0, X], y0_ivp, method='RK45', max_step=h, dense_output=True)
x_ivp = np.linspace(x0, X, 500)

y_ivp = sol.sol(x)[0]
data = []
for k in range(0, 6):
    data.append([k, x[k], y[k], y_ivp[k], abs(y[k] - y_ivp[k])])
print(tabulate(data, headers=['k', 'аргумент x(k)', 'значение y(k)', 'точное решение y(k)', 'погрешность y(k)'], tablefmt='grid', floatfmt=".10f"))

y_ivp = sol.sol(x_ivp)[0]
plt.plot(x, y, 'o-', color='blue', label=f'Runge-Kutta h={h}')
plt.plot(x_ivp, y_ivp, '-', color='red', label='Точное решение')
plt.legend()
plt.grid(True)
plt.suptitle("Task 1")
plt.show()



#################TASK2#################
print("\n############TASK2############")

x0 = 1
y0 = 0
z0 = 1.3
X = 1.5
y0_ivp = [y0, z0]
eps = 1e-4

def f1(x, y, z):
    return (x**3) * z + (2 * y)/x

def f2(x, y, z):
    return x + z / (1 + x)

# Для встроенного вычисления
def F(t, Y):
    y, z = Y
    return [ (t**3)*z + 2*y/t,
             t + z/(1+t) ]

def runge_kutta_system(f1,f2, x0:float, y0:float, z0:float, h:float, X:float):
    x = [x0]
    y = [y0]
    z = [z0]
    N = int((X - x0) / h)
    for i in range(N):
        K1y = h * f1(x0,y0,z0)
        K1z = h * f2(x0,y0,z0)
        K2y = h * f1(x0 + h/2, y0 + K1y/2, z0 + K1z/2)
        K2z = h * f2(x0 + h/2, y0 + K1y/2, z0 + K1z/2)
        K3y = h * f1(x0 + h/2, y0 + K2y/2, z0 + K2z/2)
        K3z = h * f2(x0 + h/2, y0 + K2y/2, z0 + K2z/2)
        K4y = h * f1(x0 + h, y0 + K3y, z0 + K3z)
        K4z = h * f2(x0 + h, y0 + K3y, z0 + K3z)
        dy = (K1y + 2*K2y + 2*K3y + K4y)/6
        dz = (K1z + 2*K2z + 2*K3z + K4z)/6
        x0 = x0 + h
        y0 = y0 + dy
        z0 = z0 + dz
        x.append(x0)
        y.append(y0)
        z.append(z0)
    return np.array(x), np.array(y), np.array(z)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
axes = axes.flatten()

# Решение через встроенную функцию
sol = solve_ivp(F, [x0, X], y0_ivp, method='RK45', max_step=h/1000, dense_output=True)
x_ivp = np.linspace(x0, X, 500)
y_ivp, z_ivp = sol.sol(x_ivp)
axes[0].plot(x_ivp, z_ivp, '-', color='green', label='z(x) точное решение')
axes[1].plot(x_ivp, z_ivp, '-', color='green', label='z(x) точное решение')
axes[0].plot(x_ivp, y_ivp, '-', color='black', label='y(x) точное решение')
axes[1].plot(x_ivp, y_ivp, '-', color='black', label='y(x) точное решение')

x, y, z = runge_kutta_system(f1,f2, x0, y0, z0, h, X)

print("Таблица значений")
data = []
for k in range(0, 6):
    data.append([k, x[k], y[k], z[k]])
print(tabulate(data, headers=['k', 'аргумент x(k)', 'значение y(k)', 'значение z(k)'], tablefmt='grid', floatfmt=".6f"))

print("Таблица погрешностей")
y_ivp, z_ivp = sol.sol(x)
data = []
for k in range(0, 6):
    data.append([k, x[k], abs(y[k] - y_ivp[k]), abs(z[k] - z_ivp[k])])
print(tabulate(data, headers=['k', 'аргумент x(k)', 'погрешность y(k)', 'погрешность z(k)'], tablefmt='grid', floatfmt=".10f"))

print("Погрешность y(1.5) = " + str(abs(y_ivp[-1] - y[-1])))
print("Погрешность z(1.5) = " + str(abs(z_ivp[-1] - z[-1])))
axes[0].plot(x, y, 'o--', color='blue', label=f'y(x), h={h}')
axes[0].plot(x, z, 'o--', color='red', label=f'z(x), h={h}')

# График с h = h/2
x, y, z = runge_kutta_system(f1,f2, x0, y0, z0, h/2, X)
axes[1].plot(x, y, 'o--', color='blue', label=f'y(x), h={h/2}')
axes[1].plot(x, z, 'o--', color='red', label=f'z(x), h={h/2}')

for i in range(2):
    axes[i].legend()
    axes[i].grid(True)

plt.suptitle("Task 2")
plt.show()
