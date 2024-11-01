import torch.nn as nn
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


DEVICE = "cpu"
IMAGE_SIZE = 256
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    input_image = transform(img)
    return input_image


def save_image(tensor, save_path):
    tensor = tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(tensor)
    img.save(save_path)
    print(f"Generated image saved to {save_path}")

def inference(generator_model_path, input_image_path, output_image_path):
    # Load generator model and set it to evaluation mode
    generator = Generator(in_channels=3, features=64).to(DEVICE)
    
    # Load and update the model state dictionary
    state_dict = torch.load(generator_model_path, map_location=DEVICE)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    generator.load_state_dict(new_state_dict)
    generator.eval()

    # Load and prepare the input image
    satellite_image = load_image(input_image_path).unsqueeze(0).to(DEVICE)

    # Generate output
    with torch.no_grad():
        fake_map_image = generator(satellite_image)
    
    # Save the generated image
    save_image(fake_map_image, output_image_path)

# Example usage:
# inference("path_to_generator.pth", "path_to_input_image.jpg", "path_to_save_output_image.jpg")
for i in range( 20, 701, 20):
    inference(f"/scratch/ep23btech11012.phy.iith/Pixtopix/anime/save_gen/generator_epoch_{i}.pth", 
              "/scratch/ep23btech11012.phy.iith/Pixtopix/anime/temp/Kripalu sketch - Kripalu Vipul Sonar.jpg", 
              f"/scratch/ep23btech11012.phy.iith/Pixtopix/anime/output/output3/image_{i}.jpg")