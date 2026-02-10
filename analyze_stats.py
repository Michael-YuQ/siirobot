"""Quick analysis of training_stats.json"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/stats_full.json"

with open(path) as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")
print(f"\nFirst 3 entries:")
for e in data[:3]:
    print(f"  iter={e.get('iter')} reward={e.get('reward')} solver_reward={e.get('solver_reward', 'N/A')} v_loss={e.get('v_loss', 'N/A'):.6f} source={e.get('source', 'N/A')}")

print(f"\nLast 3 entries:")
for e in data[-3:]:
    print(f"  iter={e.get('iter')} reward={e.get('reward')} solver_reward={e.get('solver_reward', 'N/A')} v_loss={e.get('v_loss', 'N/A'):.6f} source={e.get('source', 'N/A')}")

# Stats
rewards = [e.get('reward', 0) for e in data]
solver_rewards = [e.get('solver_reward', 0) for e in data if 'solver_reward' in e]
v_losses = [e.get('v_loss', 0) for e in data if 'v_loss' in e]

print(f"\nReward stats:")
print(f"  min={min(rewards):.4f} max={max(rewards):.4f} mean={sum(rewards)/len(rewards):.4f}")

if solver_rewards:
    print(f"\nSolver reward stats:")
    print(f"  min={min(solver_rewards):.4f} max={max(solver_rewards):.4f} mean={sum(solver_rewards)/len(solver_rewards):.4f}")

if v_losses:
    print(f"\nValue loss stats:")
    print(f"  first 10 mean: {sum(v_losses[:10])/10:.6f}")
    print(f"  last 10 mean:  {sum(v_losses[-10:])/10:.6f}")

# Check sources
sources = {}
for e in data:
    s = e.get('source', 'N/A')
    sources[s] = sources.get(s, 0) + 1
print(f"\nSource distribution: {sources}")

# Sample every 200 iters to see progression
print(f"\nProgression (every 200 iters):")
for i in range(0, len(data), 200):
    e = data[i]
    sr = e.get('solver_reward', 'N/A')
    sr_str = f"{sr:.4f}" if isinstance(sr, float) else sr
    print(f"  iter={e.get('iter'):4d} solver_rew={sr_str} v_loss={e.get('v_loss', 0):.6f}")
