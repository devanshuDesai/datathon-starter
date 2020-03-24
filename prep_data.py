import pandas as pd
import os
from sklearn.model_selection import train_test_split as tts

train = pd.read_csv('train.csv')
X_train, X_val, _, _ = tts(train, train.breedID, train_size=0.85)
os.system('mv train.csv orig.csv')
X_train.to_csv('train.csv', index=False)
X_val.to_csv('val.csv', index=False)

X_val.fname.apply(lambda x: os.system(f'mv image/train/{x}.jpg image/val/'));

# Fix 4-channel images
for path in glob('image/*/*'):
    img_arr = plt.imread(path)
    if (img_arr.shape[2]!=3):
        plt.imsave(path, img_arr[:,:,[0,2,3]])
