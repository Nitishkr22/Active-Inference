from tiny_env import TinyIndoorEnv


if __name__ == "__main__":
    env = TinyIndoorEnv()
    result = env.reset()
    import matplotlib.pyplot as plt

    print("Initial info:", result.info)
    env.render_topdown_ascii()
    print("Obs shape:", result.observation.shape)
    img = env._render_first_person()
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title("First-person view")
    plt.axis("off")
    plt.show()
    actions = [0, 0, 2, 0, 0, 1, 0, 0,2,0,0,0,0,0,1,0,0,0]  # forward, forward, right, ...
    for a in actions:
        result = env.step(a)
        print("\nAction:", env.ACTION_NAMES[a])
        print("Info:", result.info)
        print("Reward:", result.reward, "Done:", result.done)
        env.render_topdown_ascii()
        img = env._render_first_person()

        # plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        plt.title("First-person view")
        plt.axis("off")
        plt.show()
        # plt.axis("off")
        # plt.pause(0.01)  # pause to update the plot
        if result.done:
            break