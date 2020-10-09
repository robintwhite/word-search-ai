from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

data_dir = Path('data')
paths = list(data_dir.glob('*/*.png'))

labels = []

for imagePath in paths:
    # load image, preprocess, and store
    label = imagePath.parent.stem
    labels.append(label)

letter_counts = Counter(labels)
df = pd.DataFrame.from_dict(letter_counts, orient='index')
df.plot(kind='bar')
plt.show()
