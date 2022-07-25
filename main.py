import tqdm
import numpy as np
from model.Unet import Unet
import torch, glob
from torch import nn
from dataloader import CustomDataset, gen_test_img
from torch.optim import Adam
import torchvision.utils as vutils

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resume = True

train_data = glob.glob("data/train/*")
test_data = glob.glob("data/test/*")

train_dataset = CustomDataset(train_data)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

unet = Unet().to(device)
resume_epoch = 0

optimizer = Adam(unet.parameters(), lr=LR, betas=(0.5, 0.999))
loss = nn.MSELoss()
test_input = gen_test_img(test_data).to(device)
print(len(test_data), len(test_input))
loss_num = 99999999.0
for epoch in range(resume_epoch, EPOCHS):
    print('Epoch: ',epoch)
    errors = []
    for (input, label) in tqdm.tqdm(trainloader):
        unet.train()
        optimizer.zero_grad()
        input = input.to(device)
        label = label.to(device)

        output = unet(input)
        error = loss(output, label)*2
        error.backward()
        optimizer.step()
        errors.append(error.cpu().detach().numpy())

    loss_tmp = np.mean(errors)
    info = "Epoch {} MSELoss {}\n".format(epoch, loss_tmp)
    with open("logger.txt", "a") as f:
        f.write(info)
    if loss_num > loss_tmp:
        torch.save(unet, "ckpt/best.pth")
        loss_num = loss_tmp
        unet.eval()
        for idx, j in enumerate(test_input):
            j = j.unsqueeze(0)
            name = test_data[idx].split("/")[-1]
            vutils.save_image(unet(j),'results/'+name, normalize=True)
    torch.save(unet, "ckpt/last.pth")
