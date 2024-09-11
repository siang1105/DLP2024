from diffusers import UNet2DModel
import torch
import torch.nn as nn

# This code is inspired by the Hugging Face Diffusers tutorial:
# https://huggingface.co/docs/diffusers/tutorials/basic_training
class conditionalDDPM(nn.Module):
    def __init__(self, num_classes = 24, dim = 512):
        super().__init__()

        # Initialize a UNet2DModel, a generic U-Net model from the diffusers library for denoising and image generation
        self.ddpm = UNet2DModel(
            sample_size = 64, # size of input images 64*64
            in_channels = 3, # RGB
            out_channels = 3,
            layers_per_block = 2, # Number of ResNet layers in each U-Net block
            block_out_channels = [128, 128, 256, 256, 512, 512], # the number of output channels for each UNet block

            # Standard 2D downsampling block
            down_block_types=[
                "DownBlock2D", 
                "DownBlock2D", 
                "DownBlock2D", 
                "DownBlock2D", 
                "AttnDownBlock2D", 
                "DownBlock2D"
            ],

            # Standard 2D upsampling block
            up_block_types=[
                "UpBlock2D", 
                "AttnUpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D"
            ],

            # Use an identity function for class embedding (no transformation)
            class_embed_type="identity",
        )

        # Define a fully connected layer to embed class labels into vectors matching the model dimension
        self.class_embedding = nn.Linear(num_classes, dim)

    def forward(self, x, t, label):

        # Convert class labels into embedding vectors matching the model dimensions
        class_embed = self.class_embedding(label)

        # Return the denoised image
        return self.ddpm(x, t, class_embed).sample


if __name__ == "__main__":
    model = conditionalDDPM()
    # print(model)
    # print(model(torch.randn(1, 3, 64, 64), 10, torch.randint(0, 1, (1, 24), dtype=torch.float)).shape)