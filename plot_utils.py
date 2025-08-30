# import utils
import os
import matplotlib.pyplot as plt

SAVEPATH = "Media"
os.makedirs(SAVEPATH, exist_ok=True)

def showsave(title="Figure"):
    name = os.path.join(SAVEPATH, f"{title}.png")

    if os.path.exists(name):
        plt.show()
        print(f"File {name} already exists. Skipping.")
        return

    plt.gcf().savefig(name, dpi=400, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to: {name}")