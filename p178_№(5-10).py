import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

x0 = [0, 1, 0, 0, 1, 2]
b = [1, 1.5, 0.5, 1, 2, 2.5]
h = [0.1, 0.05, 0.05, 0.1, 0.1, 0.05]
y0 = [1, 2, math.e, 0, 0, 2]

def euler(f, x0:float, y0:float, h:float, X:float):
    x = [x0]
    y = [y0]
    N = int((X - x0) / h)
    for i in range(N):
        y0 = y0 + h * f(x0, y0)
        x0 = x0 + h
        x.append(x0)
        y.append(y0)
    return np.array(x), np.array(y)

f = [
    lambda x, y: (x + y) / (y - x),                                # 5
    lambda x, y: (6 - x**2 * y**2) / (-x**2),                      # 6
    lambda x, y: -(x*y) / (math.sqrt(1 - x**2)),                   # 7
    lambda x, y: 1 / math.cos(x) - (y * math.sin(x)) / math.cos(x),# 8
    lambda x, y: (3 * y) / (2 * x) + (3 * x * (y**(1/3))) / 2,     # 9
    lambda x, y: (2 * x * y**3) / (1 - x**2 * y**2)                # 10
]

titles = [
    r"$y' = \frac{x+y}{y-x}$",
    r"$y' = \frac{6 - x^2 y^2}{-x^2}$",
    r"$y' = -\frac{xy}{\sqrt{1-x^2}}$",
    r"$y' = \frac{1}{\cos x} - y \frac{\sin x}{\cos x}$",
    r"$y' = \frac{3}{2x}y + \frac{3}{2} x y^{1/3}$",
    r"$y' = \frac{2xy^3}{1 - x^2 y^2}$"
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten() # type: ignore

for i in range(6):
    y0_ivp = [y0[i]]
    x_euler, y_euler = euler(f[i], x0[i], y0[i], h[i], b[i])
    x_euler2, y_euler2 = euler(f[i], x0[i], y0[i], h[i] / 2, b[i])
    
    # Решение через solve_ivp
    sol = solve_ivp(f[i], [x0[i], b[i]], y0_ivp, method='RK45', max_step=h[i], dense_output=True)
    x_ivp = np.linspace(x0[i], b[i], 200)
    y_ivp = sol.sol(x_ivp)[0]

    axes[i].plot(x_euler, y_euler, 'o-', color='blue', label=f'Euler h={h[i]}')
    axes[i].plot(x_euler2, y_euler2, 's-', color='green', label=f'Euler h={h[i] / 2}')
    axes[i].plot(x_ivp, y_ivp, '-', color='red', label='RK45')
    
    axes[i].legend()
    axes[i].grid(True)
    axes[i].set_xlabel("x")
    axes[i].set_ylabel("y")
    axes[i].set_title(titles[i])

plt.tight_layout()
plt.show()
