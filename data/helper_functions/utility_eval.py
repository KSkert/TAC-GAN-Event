import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

file_a = pd.read_csv('../GAN_Embeddings_RNN_Feedback_1_epoch_metrics.csv')
file_b = pd.read_csv('../GAN_Embeddings_RNN_Feedback_2_epoch_metrics.csv') 

# Filter out epoch 0 (no generator pass means no lstm feedback)
file_a = file_a[file_a['epoch'] != 0]
file_b = file_b[file_b['epoch'] != 0]

x_a = file_a['discriminator_fake_acc']
wmape_a = file_a['generator_avg_wmape']
cov_contrib_a = file_a['generator_cov_contrib']

x_b = file_b['discriminator_fake_acc']
wmape_b = file_b['generator_avg_wmape']
cov_contrib_b = file_b['generator_cov_contrib']

# WMAPE vs Discriminator Fake Accuracy
plt.figure(figsize=(8, 5))
plt.plot(x_a, wmape_a, 'r-', label='WMAPE (Test 4)', marker='o')
plt.plot(x_b, wmape_b, 'b-', label='WMAPE (Test 5)', marker='o')

plt.xlabel('Discriminator Fake Accuracy')
plt.ylabel('Average WMAPE')
plt.title('Generator Average WMAPE vs Discriminator Fake Accuracy')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis() # decrease is positive direction
plt.tight_layout()
plt.show()

# Covariant Contribution vs Discriminator Fake Accuracy
plt.figure(figsize=(8, 5))
plt.plot(x_a, cov_contrib_a, 'r--', label='Covariant Contribution (Test 4)', marker='x')
plt.plot(x_b, cov_contrib_b, 'b--', label='Covariant Contribution (Test 5)', marker='x')

plt.xlabel('Discriminator Fake Accuracy')
plt.ylabel('Covariant Contribution')
plt.title('Generator Covariant Contribution vs Discriminator Fake Accuracy')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()  # decrease is positive direction
plt.tight_layout()
plt.show()

# WMAPE vs Covariant Contribution :
plt.figure(figsize=(8, 5))
plt.plot(cov_contrib_a, wmape_a, 'r-', label='Test 4', marker='o')
plt.plot(cov_contrib_b, wmape_b, 'b-', label='Test 5', marker='o')

plt.xlabel('Covariant Contribution')
plt.ylabel('Average WMAPE')
plt.title('Generator Average WMAPE vs Covariant Contribution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
