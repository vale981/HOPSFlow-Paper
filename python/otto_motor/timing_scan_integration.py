from timing_scan import *

ot.integrate_online_multi(models, 1000)

f, a = plt.subplots()
powers = []
for model in models:
    pu.plot_with_σ(model.t, model.total_power(), ax=a)
    #print(model.power().value, model.efficiency().value)
    powers.append(model.total_power().value.max())

plt.matshow(np.array(powers).reshape((3,3)))
