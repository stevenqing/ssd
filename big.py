import pandas as pd
import numpy as np

# Read the CSV data
df = pd.read_csv('data/Cleanup_7/CF/cf_01.csv')

# Extract the 'Step' and reward columns
step = df['Step'].values
reward = df['cleanup_0.3_0.0_0.85_0.015_cbam_causal - rollout/ep_rew_mean'].values

# Create a new array of 314 evenly spaced points
new_step = np.linspace(step[0], 20032000, 314)

# Interpolate the reward values for the new steps
new_reward = np.interp(new_step, step, reward)

# Create a new DataFrame with the interpolated data
new_df = pd.DataFrame({
    'Step': new_step,
    'Reward': new_reward
})

# Round the 'Step' column to integers
new_df['Step'] = new_df['Step'].round().astype(int)

# Save the new DataFrame to a CSV file
new_df.to_csv('data/Cleanup_7/CF/cf_01.csv', index=False)

print("Interpolation complete")