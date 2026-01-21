import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

def generate_mnist_grid():
    # Ensure directories exist
    os.makedirs('./media/data', exist_ok=True)
    os.makedirs('./media/images', exist_ok=True)

    # Load MNIST data
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./media/data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True)

    # Get one batch of images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Create a grid of images
    grid_img = torchvision.utils.make_grid(images, nrow=8, padding=2)

    # Save the image
    save_image(grid_img, './media/images/mnist_examples.png')
    print("Saved MNIST grid to ./media/images/mnist_examples.png")

if __name__ == '__main__':
    generate_mnist_grid()
