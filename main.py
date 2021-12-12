# Librairies
from __future__ import division

import numpy as np
from numpy.linalg import det
from numpy.linalg import inv
from numpy.linalg import eig
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as pyplot


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def line(x, a, b):
    return a * x + b


"""
Fonction qui creer une matrice de shape (4,3,2) et affiche ces caractéristiques
"""


def numpyFunction1():
    a = np.random.randint(0, 100, size=(4, 3, 2))
    print(a)
    print("Nombre de dimension : " + str(a.ndim))
    print("Shape : " + str(a.shape))
    print("Taille du tableau : " + str(a.size))
    print("dtype : " + str(a.dtype))
    print("itemsize : " + str(a.itemsize))
    print("data : " + str(a.data))


"""
Fonction qui crée deux matrices (3,3) et fait des opérations simples
"""


def numpyFunction2():
    matrix1 = np.random.randint(0, 8, size=(3, 3))
    matrix2 = np.random.randint(2, 10, size=(3, 3))
    matrix3 = matrix1 * matrix2
    matrix4 = matrix1.T

    print(matrix1)
    print(matrix2)
    print(matrix3)
    print(matrix4)


"""
Fonction qui crée une matrice (3,3) et fait des opérations complexes
"""


def numpyFunction3():
    matrix1 = np.random.randint(0, 8, size=(3, 3))
    a = np.array([[3, 1], [1, 2]])
    b = np.array([9, 8])

    determinantmatrix1 = det(matrix1)
    inversematrix1 = inv(matrix1)

    x = np.linalg.solve(a, b)
    D, V = eig(matrix1)  # valeurs propres et vecteurs propres

    print(determinantmatrix1)
    print(inversematrix1)
    print(x)
    print(D)
    print(V)


"""
Affiche un ensemble de point (fonction affine) et affiche de l'approche avec une courbe 
"""


def scipyFunction1():
    x = np.random.uniform(0., 100., 100)
    y = 3. * x + 2. + np.random.normal(0., 10., 100)
    plt.plot(x, y, '.')
    popt, pcov = curve_fit(line, x, y)
    e = np.repeat(10., 100)
    plt.errorbar(x, y, yerr=e, fmt="none")
    popt, pcov = curve_fit(line, x, y, sigma=e)
    plt.errorbar(x, y, yerr=e, fmt="none")
    xfine = np.linspace(0., 100., 100)  # define values to plot the function for
    plt.plot(xfine, line(xfine, popt[0], popt[1]), 'r-')
    plt.show()


"""
Affiche une image redimmensionnée se situant dans le repertoire courant
"""


def scipyFunction2():
    with Image.open("image.png") as im:
        im2 = im.resize((600, 600))
        im2.show()


"""
Création de valeur avec la loi uniform et affichage d'indicateur
"""


def scipyFunction3():
    x = stats.uniform.rvs(10, size=1000)

    print("la valeur minimum : " + str(x.min()))  # valeur min
    print("la valeur maximum : " + str(x.max()))  # valeur max
    print("la valeur moyenne " + str(x.mean()))  # valeur moyenne
    print("la variance : " + str(x.var()))  # variance


if __name__ == '__main__':
    print_hi('Debut du programme')
    numpyFunction1()
    # numpyFunction2()
    # numpyFunction3()
    # scipyFunction1()
    # scipyFunction2()
    # scipyFunction3()
    print_hi('Fin du programme')
