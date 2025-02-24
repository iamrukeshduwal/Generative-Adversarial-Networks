# Generative Adversarial Network (GAN) for Custom Dataset

## Overview
This project implements a Generative Adversarial Network (GAN) to generate images for a custom dataset containing four classes: **0, 6, A, W**. The dataset consists of **512x512 RGB images**, and the model learns to generate similar images through adversarial training.

## Model Architecture
The GAN consists of two neural networks:
1. **Generator**: Takes a random noise vector and generates an image.
2. **Discriminator**: Distinguishes real images from generated images.

The generator and discriminator are trained in a competitive setting where the generator tries to produce realistic images, and the discriminator learns to distinguish them from real images.

## Dataset Structure
The dataset is stored in the `datasets/` directory with the following structure:
```
datasets/
    ├── 0/
    ├── 6/
    ├── A/
    ├── W/
```
Each folder contains images corresponding to the respective class.

## Installation
To run the project, install the required dependencies:
```bash
pip install torch torchvision tqdm matplotlib
```

## Training the GAN
1. **Prepare Dataset**: Place your images inside the `datasets/` folder.
2. **Run Training**: Execute the training script:

3. **Monitor Training**: The training process prints loss values and visualizes generated images at regular intervals.

## Usage
### Generating Images
After training, you can generate new images using the trained model:
```python
import torch
from generator import Generator
from utils import get_noise, show_tensor_images

# Load trained model
gen = Generator(z_dim=64)
gen.load_state_dict(torch.load("generator.pth"))
gen.eval()

# Generate images
z_dim = 64  # Must match the training configuration
num_images = 10  # Number of images to generate
noise = get_noise(num_images, z_dim)
fake_images = gen(noise)
show_tensor_images(fake_images)
```

## Results
The model generates realistic images of classes **0, 6, A, and W**. Below is an example of generated images:
![Generated Samples](samples/generated_images.png)

## Troubleshooting
- **CUDA Out of Memory**: Reduce batch size or resize images before training.
- **Mode Collapse (Same Image Repeatedly Generated)**: Improve training stability by adjusting learning rate and adding noise.

## Future Improvements
- Implement **Conditional GAN (cGAN)** to generate specific classes.
- Train on a larger dataset for better generalization.

## License
This project is open-source under the MIT License.

## Contact
For questions, feel free to reach out!

Happy Coding! 🚀

