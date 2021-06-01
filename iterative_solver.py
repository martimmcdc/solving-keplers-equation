### imports
from numpy import sin

### Método iterativo
def iterative_solver(e, M, epsilon=1e-9, iter_counter=False):
    x0 = M                  # initial guess
    x = M + e*sin(x0)       # primeira iteração
    counter = 1
    while abs(x0-x)>x*epsilon: # defini o erro como erro relativo entre iterações sucessivas
        x0 = x                 # atualizar último valor calculado
        x = M + e*sin(x0)   # calcular seguinte
        counter += 1

    if iter_counter:
        return x,counter

    return x
