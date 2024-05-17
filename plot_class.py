import matplotlib.pyplot as plt
import torch

def plot1(l, l_val, output_val, v_val):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the first plot on the first axis
    axes[0].plot(l)
    axes[0].set_title("Loss for the Training Set")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")


    # Plot the second plot on the second axis
    axes[1].plot(l_val)
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

def plot2(output_val, v_val):
    # Compute loss for validation set
    loss_val = torch.abs(output_val - v_val)

    # Plot the loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_val.detach().numpy(), label='Validation Loss', marker='o')
    #plt.plot(output_val.detach().numpy(), label = 'Neural network')
    #plt.plot(v_val, label = 'Value')
    plt.title('Validation Set Absolute Error')
    plt.xlabel('Random time and space values (Each batch)')
    plt.ylabel('Absolute error')
    plt.grid(True)
    plt.show()

def plot3 (losses, losses_val, output_val, a_val):
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

    # Plot the first scatter plot on the first axis
    axes[0].scatter(a_val[:, 0], a_val[:, 1])
    axes[0].set_title("Calculation of a* as in Exercise 1.1 for the Validation Set")
    axes[0].set_xlabel("a_1")
    axes[0].set_ylabel("a_2")


    # Plot the second scatter plot on the second axis
    axes[1].scatter(output_val[:, 0], output_val[:, 1])
    axes[1].set_title("Neural Network Output of a* for the Validation Set")
    axes[1].set_xlabel("a_1")
    axes[1].set_ylabel("a_2")

    # Show the plots
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()

def plot4(output_val, a_val):
    # Compute loss
    loss_val = torch.abs(output_val - a_val)

    # Plot the loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_val[:,0].detach().numpy(), label='Absolute error for a*_1', marker='o')
    plt.plot(loss_val[:,1].detach().numpy(), label='Absolute error for a*_2', marker='o')
    plt.title('Validation Set Absolute Error')
    plt.xlabel('Random time and space values (Each batch)')
    plt.ylabel('Absolute error')
    plt.legend()
    plt.grid(True)
    plt.show() 