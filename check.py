import os, cv2
from PIL import Image
for file in os.listdir("data/test"):
    img = cv2.imread("data/test/" + file)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray= img[:, :, 2]
    gray = Image.fromarray(gray, mode="L")
    gray.save("data/test_gray/" + file[:-4] + ".png",  "PNG")