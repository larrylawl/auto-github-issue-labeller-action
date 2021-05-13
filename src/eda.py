import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()
ROOT = os.environ.get("ROOT")

labels = ['URLs', 'Code', 'Logs']
seen = [0.856, 0.460, 0.223]
unseen = [0.800, 0.420, 0.198]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, seen, width, label='seen repos')
rects2 = ax.bar(x + width/2, unseen, width, label='unseen repos')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('Proportion of Machine information in seen and unseen repos.')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.savefig(f"{ROOT}/results/eda.png")
