import numpy as np

F = 200
T = 300

# low background noise
S = 0.15 * np.random.randn(F, T)

# horizontal persistent band
S[92:98, :] += 1.8

# big central burst
t = np.arange(T)
f = np.arange(F)
Tg, Fg = np.meshgrid(t, f)

burst = 7.0 * np.exp(-((Tg - 150)**2 / (2 * 28**2) + (Fg - 105)**2 / (2 * 18**2)))
S += burst

# a couple of vertical-ish burst streaks for realism
S[70:140, 145:148] += 2.5
S[80:130, 160:162] += 1.8

# keep values nonnegative
S = np.clip(S, 0, None)

np.save("spec_slice.npy", S)
print("saved spec_slice.npy with shape", S.shape)