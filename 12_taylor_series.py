import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = sp.symbols('x')

# -------------------------------------------------------
# Helper: Convert sympy expression → numpy function
# with safe evaluation (vector support)
# -------------------------------------------------------
def safe_eval(expr, x_vals):
    f = sp.lambdify(x, expr, "numpy")
    return np.array([f(val) for val in x_vals], dtype=float)


# -------------------------------------------------------
# TASK 1 : log(x) Taylor around a = 1
# -------------------------------------------------------
f = sp.log(x)
a = 1
ns = [2, 4, 6, 8]

x_plot = np.linspace(0.1, 3, 400)
y_true = np.log(x_plot)

plt.figure()
plt.plot(x_plot, y_true, label="True log(x)", linewidth=2)

for n in ns:
    series = f.series(x, a, n).removeO()
    y_approx = safe_eval(series, x_plot)
    plt.plot(x_plot, y_approx, label=f"n={n}")

plt.title("Task 1: log(x) Taylor approx around a=1")
plt.legend()
plt.grid()
plt.show()


# -------------------------------------------------------
# TASK 2 : e^x Maclaurin
# -------------------------------------------------------
f = sp.exp(x)
ns = [2, 4, 6, 8, 10]

x_plot = np.linspace(-3, 3, 400)
y_true = np.exp(x_plot)

plt.figure()
plt.plot(x_plot, y_true, label="True e^x", linewidth=2)

for n in ns:
    series = f.series(x, 0, n).removeO()
    y_approx = safe_eval(series, x_plot)
    plt.plot(x_plot, y_approx, label=f"n={n}")

plt.title("Task 2: e^x Maclaurin")
plt.legend()
plt.grid()
plt.show()


# -------------------------------------------------------
# TASK 3 : sin(x) Maclaurin
# -------------------------------------------------------
f = sp.sin(x)
ns = [2, 4, 6, 10]

x_plot = np.linspace(-10, 10, 400)
y_true = np.sin(x_plot)

plt.figure()
plt.plot(x_plot, y_true, label="True sin(x)", linewidth=2)

for n in ns:
    series = f.series(x, 0, n).removeO()
    y_approx = safe_eval(series, x_plot)
    plt.plot(x_plot, y_approx, label=f"n={n}")

plt.title("Task 3: sin(x) Maclaurin")
plt.legend()
plt.grid()
plt.show()


# -------------------------------------------------------
# TASK 4 : cos(x) Maclaurin
# -------------------------------------------------------
f = sp.cos(x)
ns = [2, 4, 6, 10]

x_plot = np.linspace(-10, 10, 400)
y_true = np.cos(x_plot)

plt.figure()
plt.plot(x_plot, y_true, label="True cos(x)", linewidth=2)

for n in ns:
    series = f.series(x, 0, n).removeO()
    y_approx = safe_eval(series, x_plot)
    plt.plot(x_plot, y_approx, label=f"n={n}")

plt.title("Task 4: cos(x) Maclaurin")
plt.legend()
plt.grid()
plt.show()


# -------------------------------------------------------
# TASK 5 : x*cos(x) Maclaurin
# -------------------------------------------------------
f = x * sp.cos(x)
ns = [2, 4, 6, 10]

x_plot = np.linspace(-10, 10, 400)
y_true = x_plot * np.cos(x_plot)

plt.figure()
plt.plot(x_plot, y_true, label="True x*cos(x)", linewidth=2)

for n in ns:
    series = f.series(x, 0, n).removeO()
    y_approx = safe_eval(series, x_plot)
    plt.plot(x_plot, y_approx, label=f"n={n}")

plt.title("Task 5: x*cos(x) Maclaurin")
plt.legend()
plt.grid()
plt.show()


# -------------------------------------------------------
# TASK 6 : cos(x^2) Maclaurin
# -------------------------------------------------------
f = sp.cos(x**2)
ns = [2, 4, 6, 10]

x_plot = np.linspace(-3, 3, 400)
y_true = np.cos(x_plot**2)

plt.figure()
plt.plot(x_plot, y_true, label="True cos(x^2)", linewidth=2)

for n in ns:
    series = f.series(x, 0, n).removeO()
    y_approx = safe_eval(series, x_plot)
    plt.plot(x_plot, y_approx, label=f"n={n}")

plt.title("Task 6: cos(x^2) Maclaurin")
plt.legend()
plt.grid()
plt.show()


# -------------------------------------------------------
# TASK 7 : x^2 * e^(-x) Maclaurin
# -------------------------------------------------------
f = x**2 * sp.exp(-x)
ns = [2, 4, 6, 8, 10]

x_plot = np.linspace(-1, 6, 400)
y_true = x_plot**2 * np.exp(-x_plot)

plt.figure()
plt.plot(x_plot, y_true, label="True x^2 * e^-x", linewidth=2)

for n in ns:
    series = f.series(x, 0, n).removeO()
    y_approx = safe_eval(series, x_plot)
    plt.plot(x_plot, y_approx, label=f"n={n}")

plt.title("Task 7: x^2 * e^-x Maclaurin")
plt.legend()
plt.grid()
plt.show()


# -------------------------------------------------------
# TASK 8 : cos²(x) Maclaurin
# -------------------------------------------------------
f = sp.cos(x)**2
ns = [2, 4, 6, 8, 10]

x_plot = np.linspace(-10, 10, 400)
y_true = np.cos(x_plot)**2

plt.figure()
plt.plot(x_plot, y_true, label="True cos²(x)", linewidth=2)

for n in ns:
    series = f.series(x, 0, n).removeO()
    y_approx = safe_eval(series, x_plot)
    plt.plot(x_plot, y_approx, label=f"n={n}")

plt.title("Task 8: cos²(x) Maclaurin")
plt.legend()
plt.grid()
plt.show()
