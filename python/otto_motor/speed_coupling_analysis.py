from speed_coupling_scan import *

import random
#powers = np.array([model.power().value for model in models])
powers = np.array([random.random() for model in models])
normalized_powers = powers - powers.min()
normalized_powers /= normalized_powers.max()
colors = [Blues(power) for power in normalized_powers]
ax1 = plt.gcf().add_subplot(111, projection='3d')

_xx, _yy = np.meshgrid(δs, τ_Is)
x, y = _xx.ravel(), _yy.ravel()
dx = (δs[1] - δs[0])
dy = (τ_Is[1] - τ_Is[0])
x -= dx /2
y -= dy /2
ax1.bar3d(x, y, np.zeros_like(powers), dx, dy, powers, color=colors)
ax1.set_xticks(δs)
ax1.set_yticks(τ_Is)

ax1.set_xlabel(r"$\delta$")
ax1.set_ylabel(r"$\tau_I$")
ax1.set_zlabel(r"$P$")
