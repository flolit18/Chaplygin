"""
    Schema de Runge-Kutta d'ordre 4, commun aux deux representations.
"""


def rk4_step(f, Y, h, g=None):
    """
        Un pas de RK4 pour Y_dot = f(Y).

        g : fonction g(Y) dont on veut l'integrale sur le pas avec les memes
            poids 1, 2, 2, 1 (la vitesse du centre r_dot), optionnelle

        Retours :
        Y_next : Y_{k+1}
        dr : increment h/6 (g_1 + 2 g_2 + 2 g_3 + g_4), None si g n'est pas donne
        states : les etats intermediaires Y^(i)
    """
    # Etape 1 : pentes p_i aux etats intermediaires Y^(i)
    Y1 = Y
    p1 = f(Y1)
    Y2 = Y + h/2 * p1
    p2 = f(Y2)
    Y3 = Y + h/2 * p2
    p3 = f(Y3)
    Y4 = Y + h * p3
    p4 = f(Y4)

    Y_next = Y + h/6 * (p1 + 2*p2 + 2*p3 + p4)
    states = [Y1, Y2, Y3, Y4]

    # r par les memes poids, g_i = g(Y^(i))
    dr = None
    if g is not None:
        gs = [g(Yi) for Yi in states]
        dr = h/6 * (gs[0] + 2*gs[1] + 2*gs[2] + gs[3])

    return Y_next, dr, states
