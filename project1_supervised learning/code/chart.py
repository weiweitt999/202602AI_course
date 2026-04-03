import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Final_Dataset_Combined.csv')

# 1. 候選人分布圓餅圖
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df['Label'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Candidate Data Composition')

# 2. 情緒分布圓餅圖
plt.subplot(1, 2, 2)
df['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ffcc99','#ffb3e6'])
plt.title('Sentiment Composition')

plt.tight_layout()
plt.show()