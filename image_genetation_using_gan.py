import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



# Define the Generator network using ConvTranspose2d (Deconvolution) for upsampling
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input is the latent_dim, going into a convolutional transpose layer
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # State: (256, 7, 7)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # State: (128, 14, 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # State: (64, 28, 28)
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()  # Output values between [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator network using convolutional layers
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: (1, 28, 28)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (64, 14, 14)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (128, 7, 7)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (256, 3, 3)
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1),
            nn.Sigmoid()  # Output probability (real/fake)
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
lr = 0.0002
batch_size = 64
latent_dim = 1  # size of the random noise vector
epochs = 5

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize generator and discriminator
generator = Generator(latent_dim=latent_dim)
discriminator = Discriminator()

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training Loop
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):

        # Training the discriminator
        real_images = real_images.to(torch.device("cpu"))
        real_images = real_images.view(-1, 1, 28, 28)  # Reshape to (batch_size, 1, 28, 28)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Discriminator loss on real images
        outputs_real = discriminator(real_images)
        d_loss_real = criterion(outputs_real, real_labels)

        # Generate fake images
        z = torch.randn(batch_size, latent_dim, 1, 1)
        fake_images = generator(z)

        # Discriminator loss on fake images
        outputs_fake = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake

        # Backprop and optimize
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Training the generator
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)  # Generator wants the fake images to be classified as real

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        # Print losses
        if i % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Generate new images
z = torch.randn(batch_size, latent_dim, 1, 1)
fake_images = generator(z)

# Convert the generated images into the correct shape for plotting
fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)

# Visualize generated images
fake_images = fake_images.detach().numpy()

# Plot images
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(fake_images[i][0], cmap='gray')
    plt.axis('off')
plt.show()
