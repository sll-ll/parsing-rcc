import numpy as np
from PIL import Image

ss = Image.open('/home/sll/Desktop/workspace/QANet/data/CIHP/Training/Human_ids/0000006.png')

Image.fromarray(np.array(ss) * 100).show()

print(np.unique(ss))