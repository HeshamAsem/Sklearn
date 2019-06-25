from sklearn.datasets import load_sample_image

china = load_sample_image('china.jpg')
china.dtype
china.shape


flower = load_sample_image('flower.jpg')
flower.dtype
flower.shape

import matplotlib.pyplot as plt
plt.imshow(china)
plt.imshow(flower)



