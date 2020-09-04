import numpy as np

def RamanujamSimultaneousEquations(a=np.array([2, 3, 16, 31, 103, 235, 674, 1669, 4526, 11595])):
    """
    This code implements the beautiful algorithm given by Ramanujam to solve a specific set of
    nonlinear simultaneous polynomial (in vectors x and y) equations. Although there is a necessary
    and sufficient condition for the existence of a solution, it is not appear to be checkable in polynomial time.
    Hence the code must be used with caution. Read more at: https://pdfs.semanticscholar.org/ae75/da0be9fb455e2c55daa5fca46ae63e6a60bd.pdf

    :param a: The RHS of the equations given as a vector. Default set to the example in Ramanujam's paper.
    :return: Two vectors x, y representing the solution.
    """
    if (len(a) % 2 != 0):
        print('There must be an even number of equations!')
        return -1, -1, -1

    n = int(len(a) / 2)

    Z = np.zeros((n, n))
    for k in range(n):
        Z[k, :] = np.flip(a[k:n + k])

    z = -a[n:2 * n]

    B = np.linalg.solve(Z,z)

    Z = np.zeros((n, n))
    for k in range(n):
        Z[k, 0:k + 1] = np.flip(a[0:k + 1])

    A = Z @ np.hstack((1, B[0:n - 1]))

    roots = np.roots(np.hstack((np.flip(B), 1)))

    if (np.linalg.norm(np.imag(roots)) != 0):
        print('There does not exist a solution!')
        return -1, -1, -1

    y = 1 / roots

    x = np.zeros(n)
    for k in range(n):
        den = np.polyval(np.flip(np.polynomial.polynomial.polyfromroots(np.delete(roots, k))), roots[k])
        num = np.polyval(np.flip(A), roots[k])
        x[k] = y[k] * num / den

    Y = np.transpose(np.vander(y, increasing=True, N=len(a)))
    diff_norm = np.linalg.norm(a - Y @ x)

    return x, y, diff_norm


