# Bug Analysis: Teleop Crashes & Heuristic Spinning

## Issue 1: Teleop "Crashes" are Actually Obstacle Collisions

### What's Happening
- User thinks they're landing but the episode ends with a "crash"
- Analyzing episode `episode_20260324_170136_628489_teleop.pkl`:
  - Final position: (-0.26, 0.39) - **still 0.39 units above ground!**
  - Final status: `collision=True`, `crashed=False`
  - The lander hit a **circular obstacle**, not the ground

### Root Cause
The circular obstacles are not visually prominent enough. When descending, the user doesn't see them clearly and collides with them thinking they're landing.

### Proposed Fixes
1. **Make obstacles more visible** - increase their visual prominence (color, outline, etc.)
2. **Add collision warning indicators** - show lidar rays that detect nearby obstacles
3. **Improve termination messaging** - clearly indicate "OBSTACLE HIT" vs "CRASHED INTO TERRAIN"

## Issue 2: Heuristic Controller Spinning/Oscillating

### What's Happening
- Heuristic controller shows high-frequency oscillations in side engine control
- Angular velocity spikes briefly to 1.3 rad/s
- Lander fails to stabilize angle (ends at 41 degrees instead of upright)
- Actions show constant back-and-forth corrections

### Observation Format Comparison

**Old Environment (Box2D-based):**
```
Index:  0       1       2     3     4     5           6           7            8         9-16
Value:  pos_x   pos_y   vel_x vel_y angle angular_vel leg1_contact leg2_contact helipad_x lidar...
```

**New Environment (KTO-based):**
```
Index:  0  1  2     3   4   5     6-13    14
Value:  x  y  theta vx  vy  omega lidar   pad_x
```

**Old Heuristic Controller Logic:**
```python
angle_targ = s[0] * 0.5 + s[2] * 1.0  # x * 0.5 + vx * 1.0
angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0  # (angle_targ - angle) - angular_vel
hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5  # (hover_targ - y) - vy

if s[6] or s[7]:  # Check leg contact
    angle_todo = 0
    hover_todo = -s[3] * 0.5  # Just reduce vy
```

**Current Heuristic Controller:**
```python
x, y, theta, vx, vy, omega = s[:6]  # CORRECT extraction
angle_targ = x * 0.5 + vx * 1.0      # CORRECT: x * 0.5 + vx * 1.0
angle_todo = (angle_targ - theta) * 0.5 - omega * 1.0  # CORRECT
hover_todo = (hover_targ - y) * 0.5 - vy * 0.5  # CORRECT

if y < 3.5:  # Near ground - NO LEG CONTACT CHECK
    angle_todo = 0
    hover_todo = -vy * 0.5
```

### Root Causes

1. **Missing Leg Contact Info**: The KTO environment doesn't provide leg contact in the observation (it's not part of the simplified physics). The heuristic tries to compensate by checking `y < 3.5` (near ground), but this isn't the same thing.

2. **Physics Model Mismatch**: The KTO physics is different from Box2D:
   - KTO uses simplified 2D rocket equations
   - Box2D has more realistic collision detection, friction, damping
   - The PID gains (0.5, 1.0, etc.) were tuned for Box2D dynamics

3. **No Damping/Friction**: The KTO physics has no inherent damping, so once the lander starts rotating, the controller has to work harder to stop it, leading to oscillations.

### Proposed Fixes

**Option A: Tune PID Gains for KTO Physics**
- The current gains (0.5, 1.0, 20) were tuned for Box2D
- Need to retune for KTO physics, probably with:
  - Lower derivative gain on angle control (reduce oscillation)
  - Add more damping term for angular velocity
  - Adjust action scaling (currently `* 20`)

**Option B: Add Damping to KTO Physics**
- Add angular velocity damping: `omega_dot -= DAMPING_COEFF * omega`
- Add velocity damping: `vx_dot -= DAMPING_COEFF * vx`, `vy_dot -= DAMPING_COEFF * vy`
- This would make the physics more stable and closer to Box2D behavior

**Option C: Improve Ground Contact Detection**
- Add actual ground/terrain collision detection to KTO environment
- Provide leg contact boolean in observation
- This would allow the heuristic to know when to stop trying to correct angle

## Recommended Action Plan

1. **For Teleop Issue**:
   - Add visual indicators for obstacles (brighter colors, warning zones)
   - Add lidar ray visualization in real-time
   - Improve collision feedback

2. **For Heuristic Issue**:
   - Start with Option A (tune PID gains) as it requires no physics changes:
     - Reduce angle_todo gain from 20 to maybe 10-15
     - Reduce derivative term on omega from 1.0 to 0.5
     - Test and iterate
   - If that doesn't work well, add Option B (damping) to physics
   - Option C would require more significant changes to the environment

## Test Cases Needing Analysis
- `episode_20260324_171030_169302_heuristic.pkl` - Shows spinning/oscillation
- `episode_20260324_170136_628489_teleop.pkl` - Shows obstacle collision mistaken for landing
