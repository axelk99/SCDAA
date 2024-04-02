
import matplotlib.pyplot as plt
def plot1(losses, losses_val, output_val, v_val):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the first plot on the first axis
    axes[0].plot(losses)
    axes[0].set_title("Loss for the Training Set")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")


    # Plot the second plot on the second axis
    axes[1].plot(losses_val)
    axes[1].set_title("Loss for the Validation Set")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")


    # Show the plots
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()

    output_val = output_val.detach()
    # Create a figure and axis objects
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the first plot on the first axis
    axes[0].plot(v_val)
    axes[0].set_title("Calculation of v as in Exercise 1.1 for the Validation Set")
    axes[0].set_xlabel("Validation Set Values of x and t")
    axes[0].set_ylabel("Value function")


    # Plot the second plot on the second axis
    axes[1].plot(output_val)
    axes[1].set_title("Neural Network Output of v for the Validation Set")
    axes[1].set_xlabel("Validation Set Values of x and t")
    axes[1].set_ylabel("Value function")


    # Show the plots
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()