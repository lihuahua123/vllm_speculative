import numpy as np
import re
import matplotlib.pyplot as plt
data1 = np.load("tpots_org.npy")
data2 = np.load("tpots_ngram.npy")
data3 = np.load("tpots_ngram2.npy")
# Create a scatter plot comparing data1 and data2
plt.figure(figsize=(10, 5))
plt.scatter(range(len(data1)), data1, label='Original', alpha=0.5)
plt.scatter(range(len(data2)), data2, label='Ngram', alpha=0.5)
plt.scatter(range(len(data3)), data3, label='Ngram_no_speculative', alpha=0.5)
plt.title('Comparison of Original vs Ngram TPOTs')
plt.xlabel('Request Index') 
plt.ylabel('TPOT Value')
plt.legend()
plt.grid(True)
plt.savefig('./figs/tpot_comparison.png')

