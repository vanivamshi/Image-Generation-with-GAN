# Image-Generation-with-GAN

Generative Adversarial Network (GAN) consists of two neural networks: a generator and a discriminator. They are set up in a competitive framework, where they both learn and improve through adversarial training. Here, MNIST dataset is used for training.

Generator: This network's goal is to produce data that mimics real data (e.g., generating realistic images or sound). It starts by taking random noise as input and tries to generate fake data that looks real.

Discriminator: The role of this network is to distinguish between real data (from the actual dataset) and fake data (produced by the generator). It outputs a probability indicating whether the input is real or fake.
