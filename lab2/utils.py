import matplotlib.pyplot as plt

def plot_loss_accuracy(train_losses, train_accuracies, save_path=None, interval=50):
    epochs = range(1, len(train_losses) + 1)
    sampled_epochs = [e for i, e in enumerate(epochs) if (i+1) % interval == 0]
    sampled_train_losses = [loss for i, loss in enumerate(train_losses) if (i+1) % interval == 0]
    sampled_train_accuracies = [acc for i, acc in enumerate(train_accuracies) if (i+1) % interval == 0]

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sampled_epochs, sampled_train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sampled_epochs, sampled_train_accuracies, 'b-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    if save_path:
        plt.suptitle(f'{save_path}_training')
        plt.savefig(f"{save_path}_training.png")

    plt.show()