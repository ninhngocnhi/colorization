import matplotlib.pyplot as plt
loss = []
epoch = []
with open ("logger.txt", "r+") as f:
    for line in f.readlines():
        x, epo, y, los = line.strip().split(" ")
        loss.append(float(los))
        epoch.append(float(epo))
num_epoch = len(epoch)

fig = plt.figure()
ax1 = fig.add_subplot(111, title="loss")
ax1.plot(epoch, loss, 'b', label='train_loss')
plt.savefig('result.png')
plt.legend()
plt.show()