
import matplotlib.pyplot as plt
import re

log_data = """
Start Training: 1000 images, 10 epochs
Epoch 0: 0%| | 0/1000 [00:00<?, ?it/s]
Step 0: Total Loss=0.511115, MSE=0.508874, Shape=0.224084
Step 100: Total Loss=0.911414, MSE=0.910423, Shape=0.099171
Step 200: Total Loss=0.002354, MSE=0.000000, Shape=0.235395
Step 300: Total Loss=0.973397, MSE=0.972989, Shape=0.040801
Step 400: Total Loss=0.989026, MSE=0.988587, Shape=0.043930
Step 500: Total Loss=0.787194, MSE=0.783358, Shape=0.383526
Step 600: Total Loss=0.993790, MSE=0.993263, Shape=0.052674
Step 700: Total Loss=0.481496, MSE=0.478828, Shape=0.266859
Step 800: Total Loss=1.017442, MSE=1.017362, Shape=0.007991
Step 900: Total Loss=0.584502, MSE=0.582903, Shape=0.159859
Step 1000: Loss=0.6079 (MSE: 0.7928, Shape: 0.1140)

Epoch 1:
Step 0: Total Loss=0.442983, MSE=0.441769, Shape=0.121398
Step 100: Total Loss=0.446012, MSE=0.444482, Shape=0.152971
Step 200: Total Loss=0.880878, MSE=0.880157, Shape=0.072141
Step 300: Total Loss=0.843595, MSE=0.842812, Shape=0.078342
Step 400: Total Loss=0.906134, MSE=0.905527, Shape=0.060689
Step 500: Total Loss=0.450150, MSE=0.449545, Shape=0.060494
Step 600: Total Loss=0.001954, MSE=0.000000, Shape=0.195359
Step 700: Total Loss=1.016348, MSE=1.016026, Shape=0.032241
Step 800: Total Loss=0.699068, MSE=0.696642, Shape=0.242592
Step 900: Total Loss=0.663304, MSE=0.661695, Shape=0.160845

Epoch 2:
Step 0: Total Loss=1.000042, MSE=0.999978, Shape=0.006358
Step 100: Total Loss=0.001072, MSE=0.000000, Shape=0.107151
Step 200: Total Loss=0.001137, MSE=0.000000, Shape=0.113664
Step 300: Total Loss=1.034709, MSE=1.034588, Shape=0.012185
Step 400: Total Loss=0.752175, MSE=0.751403, Shape=0.077178
Step 500: Total Loss=0.000679, MSE=0.000000, Shape=0.067866
Step 600: Total Loss=0.482453, MSE=0.479733, Shape=0.271940
Step 700: Total Loss=0.564006, MSE=0.561473, Shape=0.253314
Step 800: Total Loss=0.001040, MSE=0.000000, Shape=0.103980
Step 900: Total Loss=0.705939, MSE=0.703758, Shape=0.218118

Epoch 3:
Step 0: Total Loss=0.001436, MSE=0.000000, Shape=0.143632
Step 100: Total Loss=0.673745, MSE=0.672962, Shape=0.078318
Step 200: Total Loss=0.867052, MSE=0.865281, Shape=0.177102
Step 300: Total Loss=0.753529, MSE=0.752244, Shape=0.128536
Step 400: Total Loss=0.001647, MSE=0.000000, Shape=0.164669
Step 500: Total Loss=0.541772, MSE=0.540441, Shape=0.133164
Step 600: Total Loss=0.468365, MSE=0.467072, Shape=0.129357
Step 700: Total Loss=0.630751, MSE=0.626818, Shape=0.393292
Step 800: Total Loss=0.591510, MSE=0.589520, Shape=0.199002
Step 900: Total Loss=0.728256, MSE=0.726907, Shape=0.134818

Epoch 4:
Step 0: Total Loss=0.545160, MSE=0.542772, Shape=0.238820
Step 100: Total Loss=0.845997, MSE=0.845077, Shape=0.092011
Step 200: Total Loss=0.799971, MSE=0.798624, Shape=0.134733
Step 300: Total Loss=0.377142, MSE=0.375548, Shape=0.159363
Step 400: Total Loss=0.942757, MSE=0.942390, Shape=0.036697
Step 500: Total Loss=1.015095, MSE=1.014171, Shape=0.092355
Step 600: Total Loss=0.420055, MSE=0.418329, Shape=0.172580
Step 700: Total Loss=0.942413, MSE=0.942066, Shape=0.034706
Step 800: Total Loss=0.675972, MSE=0.674816, Shape=0.115619
Step 900: Total Loss=0.714807, MSE=0.714389, Shape=0.041846

Epoch 5:
Step 0: Total Loss=0.648168, MSE=0.645283, Shape=0.288559
Step 100: Total Loss=0.578737, MSE=0.576809, Shape=0.192803
Step 200: Total Loss=0.492674, MSE=0.492152, Shape=0.052249
Step 300: Total Loss=0.520593, MSE=0.519531, Shape=0.106173
Step 400: Total Loss=0.000239, MSE=0.000000, Shape=0.023864
Step 500: Total Loss=1.019750, MSE=1.018746, Shape=0.100417
Step 600: Total Loss=0.000337, MSE=0.000000, Shape=0.033692
Step 700: Total Loss=0.343466, MSE=0.341865, Shape=0.160036
Step 800: Total Loss=0.684763, MSE=0.682963, Shape=0.180037
Step 900: Total Loss=1.023374, MSE=1.023286, Shape=0.008855 

Epoch 6:
Step 0: Total Loss=0.628932, MSE=0.626899, Shape=0.203366
Step 100: Total Loss=0.441743, MSE=0.439223, Shape=0.251997
Step 200: Total Loss=0.869456, MSE=0.868982, Shape=0.047436
Step 300: Total Loss=0.546734, MSE=0.543413, Shape=0.332080
Step 400: Total Loss=0.551488, MSE=0.549726, Shape=0.176193
Step 500: Total Loss=0.680932, MSE=0.678090, Shape=0.284175
Step 600: Total Loss=0.002110, MSE=0.000000, Shape=0.210979
Step 700: Total Loss=0.852379, MSE=0.851038, Shape=0.134107
Step 800: Total Loss=0.679996, MSE=0.678123, Shape=0.187307
Step 900: Total Loss=1.017858, MSE=1.017550, Shape=0.030817

Epoch 7:
Step 0: Total Loss=0.438937, MSE=0.437437, Shape=0.150032
Step 100: Total Loss=0.774543, MSE=0.772451, Shape=0.209195
Step 200: Total Loss=0.866448, MSE=0.865587, Shape=0.086156
Step 300: Total Loss=0.531243, MSE=0.528437, Shape=0.280633
Step 400: Total Loss=0.669566, MSE=0.667707, Shape=0.185852
Step 500: Total Loss=0.661452, MSE=0.660138, Shape=0.131413
Step 600: Total Loss=0.403815, MSE=0.402181, Shape=0.163481
Step 700: Total Loss=0.467667, MSE=0.466596, Shape=0.107105
Step 800: Total Loss=0.001455, MSE=0.000000, Shape=0.145499
Step 900: Total Loss=0.001469, MSE=0.000000, Shape=0.146877

Epoch 8:
Step 0: Total Loss=0.413189, MSE=0.411742, Shape=0.144743
Step 100: Total Loss=0.749776, MSE=0.748828, Shape=0.094826
Step 200: Total Loss=0.490472, MSE=0.488437, Shape=0.203439
Step 300: Total Loss=0.420742, MSE=0.418692, Shape=0.204993
Step 400: Total Loss=0.380046, MSE=0.377102, Shape=0.294398
Step 500: Total Loss=0.609807, MSE=0.608320, Shape=0.148697
Step 600: Total Loss=0.569448, MSE=0.568046, Shape=0.140220
Step 700: Total Loss=0.002202, MSE=0.000000, Shape=0.220248
Step 800: Total Loss=0.783160, MSE=0.781122, Shape=0.203824
Step 900: Total Loss=0.570726, MSE=0.569482, Shape=0.124462

Epoch 9:
Step 0: Total Loss=0.001820, MSE=0.000000, Shape=0.182007
Step 100: Total Loss=0.322966, MSE=0.320869, Shape=0.209693
Step 200: Total Loss=0.922516, MSE=0.921244, Shape=0.127134
Step 300: Total Loss=0.001892, MSE=0.000000, Shape=0.189187
Step 400: Total Loss=0.564190, MSE=0.562835, Shape=0.135475
Step 500: Total Loss=0.924561, MSE=0.923904, Shape=0.065703
Step 600: Total Loss=0.506673, MSE=0.504777, Shape=0.189687
Step 700: Total Loss=0.968432, MSE=0.967970, Shape=0.046155
Step 800: Total Loss=0.810477, MSE=0.808578, Shape=0.189952
Step 900: Total Loss=0.518382, MSE=0.515921, Shape=0.246095
"""

# Extract data
total_losses = []
mse_losses = []
shape_losses = []
steps = []

current_step_base = 0
for line in log_data.split('\n'):
    if "Step" in line:
        match = re.search(r"Step (\d+): Total Loss=([0-9.]+), MSE=([0-9.]+), Shape=([0-9.]+)", line)
        if match:
            step = int(match.group(1))
            total = float(match.group(2))
            mse = float(match.group(3))
            shape = float(match.group(4))
            
            # Since step resets per epoch, we need to add base
            # But the log has "Step 0..900" per epoch.
            # Assuming 1000 steps per epoch.
            # Wait, step number isn't resetting? No, log shows "Step 0" at start of each epoch.
            # So we need to track epoch change.
            pass
            
# Let's re-parse simply by occurrences
# Each epoch has 10 data points (0 to 900, roughly).
epoch_count = 0
step_in_epoch = 0

lines = log_data.split('\n')
cumulative_step = 0

for line in lines:
    if line.startswith("Epoch "):
        try:
            epoch_num = int(line.split(':')[0].replace("Epoch ", ""))
            current_step_base = epoch_num * 1000
        except:
            pass
            
    if "Total Loss=" in line:
        match = re.search(r"Total Loss=([0-9.]+), MSE=([0-9.]+), Shape=([0-9.]+)", line)
        if match:
            total_losses.append(float(match.group(1)))
            mse_losses.append(float(match.group(2)))
            shape_losses.append(float(match.group(3)))
            steps.append(cumulative_step)
            cumulative_step += 100 # Approx step increment based on log frequency

# Plot
plt.figure(figsize=(12, 6))
plt.plot(steps, total_losses, label='Total Loss', alpha=1.0, linewidth=2.0, color='dodgerblue', zorder=2)
plt.plot(steps, mse_losses, label='MSE Loss (Reconstruction)', alpha=1.0, linestyle=':', linewidth=2.0, color='darkorange', zorder=3)
plt.plot(steps, shape_losses, label='Shape Loss (Gradient)', alpha=0.9, color='crimson', linewidth=1.5, zorder=1)

plt.title("Stage 2 Training Loss (Braid Specialization)")
plt.xlabel("Steps (Approx)")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig("results/loss_graph_stage2.png")
print("Graph saved to results/loss_graph_stage2.png")
