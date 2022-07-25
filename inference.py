import torchvision.utils as vutils
import cv2, torch
from model.Unet import Unet
from torchvision.transforms import transforms

gray_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

unet = Unet().to(device)
unet.load_state_dict(torch.load("checkpoints/final.pth"))

gray_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

path = "img.jpg"

img = cv2.imread(path)
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray= img[:, :, 0]
gray = gray_transform(gray)
gray = torch.unsqueeze(gray, 0)
gray = gray.unsqueeze(0)
vutils.save_image(unet(gray),path[:-4] + "_predict.jpg", normalize=True)