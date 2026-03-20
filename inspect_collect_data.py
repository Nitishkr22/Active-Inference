import numpy as np

data = np.load("tiny_nav_dataset.npz")
print(data.files)
print("obs:", data["obs"].shape)
img = data["obs"][1,:,:]

print("actions:", data["actions"][1])
print("next_obs:", data["next_obs"].shape)
print("done:", data["done"].shape)
print("pos:", data["pos"].shape)
print("heading:", data["heading"][1])

# import matplotlib.pyplot as plt
# plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
# plt.title("First-person view")
# plt.axis("off")
# plt.show()