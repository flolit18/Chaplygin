"""
    Figures de la simulation : courbes 2D et vues 3D de la boule.

    Sur la boule 3D, les trois axes du corps sont dessines en couleur :
    x_b en rouge, y_b en vert, z_b en bleu. Avec R = Rz(psi) Ry(theta) Rx(phi) :
    - psi (lacet) : direction de x_b projete sur le sol,
    - theta (tangage) : inclinaison de x_b par rapport au sol,
    - phi (roulis) : rotation de y_b et z_b autour de x_b.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator

from .physics import rotation_matrix
from .controls import kinetic_energy, angular_momentum_P, relative_drift

AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")
AXIS_NAMES = ("$x_b$", "$y_b$", "$z_b$")


def _ax(ax, projection=None):
    """Cree une figure si aucun axe n'est donne."""
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection=projection)
    return ax


# ---------------------------------------------------------------------------
# Courbes 2D
# ---------------------------------------------------------------------------

def plot_trajectory(result, ax=None):
    """
        Trajectoire du centre (x, y) dans le plan
    """
    ax = _ax(ax)
    x, y = result.r[:, 0], result.r[:, 1]
    ax.plot(x, y, color="tab:blue")
    ax.plot(x[0], y[0], "o", color="tab:green", label="depart")
    ax.plot(x[-1], y[-1], "s", color="tab:red", label="arrivee")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(title="Trajectoire du centre", xlabel="x (m)", ylabel="y (m)")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_angles(result, ax=None):
    """
        phi, theta, psi en fonction du temps
    """
    ax = _ax(ax)
    for k, name in enumerate((r"$\varphi$", r"$\theta$", r"$\psi$")):
        ax.plot(result.t, result.alpha[:, k], label=name)
    ax.set(title="Angles d'Euler", xlabel="t (s)", ylabel="rad")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_reactions(sphere, result, ax=None):
    """
        lambda_1, lambda_2, |lambda| et la borne mu_s m g
    """
    ax = _ax(ax)
    lam = result.lam
    ax.plot(result.t, lam[:, 0], label=r"$\lambda_1$")
    ax.plot(result.t, lam[:, 1], label=r"$\lambda_2$")
    ax.plot(result.t, np.linalg.norm(lam, axis=1), "k", lw=1,
            label=r"$\|\lambda\|$")
    bound = sphere.mu_s * sphere.m * sphere.g
    ax.axhline(bound, color="tab:red", ls="--", label=r"$\mu_s m g$")
    ax.set(title="Reactions tangentielles", xlabel="t (s)", ylabel="N")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_invariants(sphere, result, ax=None):
    """
        Derive relative de T et de L_P (echelle log)
    """
    ax = _ax(ax)
    states = list(zip(result.alpha, result.alpha_dot))
    T = [kinetic_energy(sphere, a, ad) for a, ad in states]
    L = [angular_momentum_P(sphere, a, ad) for a, ad in states]
    # on enleve k = 0 ou la derive vaut 0 (log)
    ax.semilogy(result.t[1:], relative_drift(T)[1:], label="$T$")
    ax.semilogy(result.t[1:], relative_drift(L)[1:], label="$L_P$")
    ax.set(title="Derive relative des invariants", xlabel="t (s)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return ax


def plot_summary(sphere, result):
    """
        Les quatre courbes 2D sur une meme figure
    """
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    plot_trajectory(result, axs[0, 0])
    plot_angles(result, axs[0, 1])
    plot_reactions(sphere, result, axs[1, 0])
    plot_invariants(sphere, result, axs[1, 1])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Vues 3D
# ---------------------------------------------------------------------------

def _sphere_mesh(a, n=20):
    """Points d'une sphere de rayon a centree en 0 (repere corps)."""
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, n + 1), np.linspace(0, np.pi, n // 2 + 1))
    return a * np.stack([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])


def _draw_ball(ax, sphere, alpha, center, mesh, length=1.6):
    """Dessine la boule et ses trois axes du corps ; renvoie les artistes."""
    R = rotation_matrix(alpha)
    S = np.einsum("ij,jkl->ikl", R, mesh) + center[:, None, None]
    arts = [ax.plot_surface(*S, color="lightgray", alpha=0.3, linewidth=0)]
    for i, col in enumerate(AXIS_COLORS):
        tip = center + length * sphere.a * R[:, i]
        arts += ax.plot(*np.stack([center, tip]).T, color=col, lw=2.5)
    return arts


def _snapshot_indices(result, a, n_snap):
    """
        Instants des boules : au plus n_snap, et centres distants d'au moins
        2.2 a pour que les boules ne se chevauchent pas
    """
    r = result.r
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))])
    gap = max(2.2 * a, s[-1] / max(n_snap - 1, 1))
    idx = [0]
    for k in range(1, len(s)):
        if s[k] - s[idx[-1]] >= gap and np.linalg.norm(r[k] - r[idx[-1]]) >= 2.2 * a:
            idx.append(k)
    return np.array(idx[:n_snap])


def plot_trajectory_3d(sphere, result, n_snap=6, ax=None):
    """
        Trace du centre en 3D, avec la boule et ses axes du corps a au plus
        n_snap instants, choisis pour que les boules ne se chevauchent pas
    """
    if ax is None:
        ax = plt.figure(figsize=(8, 7)).add_subplot(projection="3d")
    a = sphere.a
    x, y = result.r[:, 0], result.r[:, 1]
    ax.plot(x, y, np.full_like(x, a), color="k", lw=1, label="centre")
    ax.plot(x, y, np.zeros_like(x), color="gray", lw=1, ls="--",
            label="point de contact")

    mesh = _sphere_mesh(a)
    for k in _snapshot_indices(result, a, n_snap):
        center = np.array([x[k], y[k], a])
        _draw_ball(ax, sphere, result.alpha[k], center, mesh)
        ax.text(x[k], y[k], 2.8*a, f"{result.t[k]:.1f} s", fontsize=8)

    # meme echelle sur les trois axes pour que la boule reste ronde
    lo = np.array([x.min() - 2*a, y.min() - 2*a, 0.0])
    hi = np.array([x.max() + 2*a, y.max() + 2*a, 2.5*a])
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_zticks([0, 2*a])
    for col, name in zip(AXIS_COLORS, AXIS_NAMES):
        ax.plot([], [], color=col, lw=2.5, label=name)
    ax.set(title="Trajectoire de la boule", xlabel="x (m)", ylabel="y (m)")
    ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_orientation(sphere, result, n=6):
    """
        Orientation de la boule a n instants regulierement espaces, chaque
        boule etant ramenee a l'origine : seuls les axes du corps bougent.
        Le repere sol (x, y, z) est en gris pour reference.
    """
    a = sphere.a
    mesh = _sphere_mesh(a)
    ks = np.linspace(0, len(result.t) - 1, n).astype(int)
    fig = plt.figure(figsize=(2.6 * n, 3.2))
    for j, k in enumerate(ks):
        ax = fig.add_subplot(1, n, j + 1, projection="3d")
        for e in np.eye(3):
            ax.plot(*np.stack([-1.6*a*e, 1.6*a*e]).T, color="lightgray", lw=0.8)
        _draw_ball(ax, sphere, result.alpha[k], np.zeros(3), mesh, length=1.5)
        lim = 1.5 * a
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(f"t = {result.t[k]:.2f} s", fontsize=10)
    for col, name in zip(AXIS_COLORS, AXIS_NAMES):
        fig.axes[0].plot([], [], color=col, lw=2.5, label=name)
    fig.legend(loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Orientation de la boule (axes du corps)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def animate_3d(sphere, result, every=20, view=0.35, interval=40):
    """
        Animation 3D de la boule qui roule, avec ses trois axes du corps.
        La camera suit la boule, la trace du point de contact est en gris.

        every : on affiche un pas sur every
        view : demi-largeur de la vue autour de la boule (m)
        A garder dans une variable : anim = animate_3d(...)[1], puis
        plt.show() ou anim.save("results/boule.gif", fps=25)
    """
    a = sphere.a
    idx = np.arange(0, len(result.t), every)
    mesh = _sphere_mesh(a)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect((1, 1, 0.5))
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    trace, = ax.plot([], [], [], color="gray", lw=1)
    for col, name in zip(AXIS_COLORS, AXIS_NAMES):
        ax.plot([], [], color=col, lw=2.5, label=name)
    ax.legend(loc="upper left", fontsize=8)
    ball = []

    def update(j):
        k = idx[j]
        center = np.array([result.r[k, 0], result.r[k, 1], a])
        for art in ball:
            art.remove()
        ball[:] = _draw_ball(ax, sphere, result.alpha[k], center, mesh)

        # trace limitee a la vue (matplotlib 3D ne coupe pas les courbes)
        tr = result.r[:k+1].copy()
        tr[np.any(np.abs(tr - center[:2]) > view, axis=1)] = np.nan
        trace.set_data_3d(tr[:, 0], tr[:, 1], np.zeros(k + 1))

        ax.set_xlim(center[0] - view, center[0] + view)
        ax.set_ylim(center[1] - view, center[1] + view)
        ax.set_zlim(0, view)
        ax.set_title(f"t = {result.t[k]:.2f} s")
        return ball + [trace]

    anim = FuncAnimation(fig, update, frames=len(idx), interval=interval)
    return fig, anim
