# Exercise 3.16 Eight-State Routing Problem using Q-Learning
import numpy as np
import pylab as plt
from Q_Utils import *

# 1. Merancang lintasan ruting 8-state (0 sampai 7)
# Hubungan antar titik/node:
points_list = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
goal = 7

# 2. Menentukan jumlah state (8 state: 0, 1, 2, 3, 4, 5, 6, 7)
MATRIX_SIZE = 8

# Tampilkan grafik lintasan
showgraph(points_list)

# 3. Membuat matriks Reward (R) dan Matriks Q (8x8)
R = createRmat(MATRIX_SIZE, points_list, goal)
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

gamma = 0.8

# 4. Pelatihan (Training) Q-Learning
scores = []
for i in range(1000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(R, current_state)
    action = sample_next_action(available_act)
    score = update(R, Q, current_state, action, gamma)
    scores.append(score)

print("Trained Q matrix:")
print(Q / np.max(Q) * 100)

# 5. Pengujian (Testing) untuk mencari jalur paling efisien dari State 0 ke Goal 7
current_state = 0
steps = [current_state]

while current_state != goal:
    next_step_index = np.where(Q[current_state, :] == np.max(Q[current_state, :]))[1]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1)[0])
    else:
        next_step_index = int(next_step_index[0])
        
    steps.append(next_step_index)
    current_state = next_step_index

# 6. Menampilkan Hasil Rute Terpendek
print("\nMost efficient path:")
print(steps)

# Visualisasi konvergensi skor
plt.plot(scores)
plt.xlabel("Iterations")
plt.ylabel("Score")
plt.title("Exercise 3.16 Training Convergence")
plt.show()