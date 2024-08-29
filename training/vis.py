import numpy as np
import matplotlib.pyplot as plt


def show_sample(
    image: np.ndarray,
    events: np.ndarray,
):
    num_events = events.shape[0]
    fig, ax = plt.subplots(1, num_events + 1, figsize=(15, 10))

    image = (image + 0.5) * 255
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")

    for i in range(num_events):
        # events are also as images
        event = events[i]
        event = (event + 1) * 255
        ax[i + 1].imshow(event)

        ax[i + 1].set_title(f"Event {i}")
        ax[i + 1].axis("off")

    plt.show()
