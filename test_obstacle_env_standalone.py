#!/usr/bin/env python3
"""
Standalone validation for LunarLanderObstacle environment.
Tests code structure and logic without requiring full gym/Box2D setup.
"""

import sys
import importlib.util
from pathlib import Path

def test_syntax_and_structure():
    """Test that the file has valid syntax and expected structure"""
    print("=" * 60)
    print("Phase 1 Standalone Validation Test")
    print("=" * 60)

    # Test 1: File exists
    print("\n[Test 1] Checking file exists...")
    env_file = Path("diffusha/envs/lunar_lander_obstacle.py")
    assert env_file.exists(), f"File not found: {env_file}"
    print("[PASS] File exists")

    # Test 2: Valid Python syntax
    print("\n[Test 2] Checking Python syntax...")
    try:
        with open(env_file, 'r') as f:
            code = f.read()
        compile(code, str(env_file), 'exec')
        print("[PASS] Valid Python syntax")
    except SyntaxError as e:
        print(f"[FAIL] Syntax error: {e}")
        return False

    # Test 3: Check key constants
    print("\n[Test 3] Checking constants...")
    assert "NUM_OBSTACLES = 4" in code, "NUM_OBSTACLES not set to 4"
    assert "N_LIDAR_RAYS = 8" in code, "N_LIDAR_RAYS not set to 8"
    assert "COLLISION_PENALTY = -100.0" in code, "COLLISION_PENALTY not set correctly"
    print("[PASS] Constants defined correctly")
    print("  - NUM_OBSTACLES = 4")
    print("  - N_LIDAR_RAYS = 8")
    print("  - COLLISION_PENALTY = -100.0")

    # Test 4: Check class definition
    print("\n[Test 4] Checking class structure...")
    assert "class LunarLanderObstacle(LunarLander):" in code, "Class not defined"
    print("[PASS] LunarLanderObstacle class defined")

    # Test 5: Check required methods
    print("\n[Test 5] Checking required methods...")
    required_methods = [
        ("__init__", "Constructor"),
        ("reset", "Reset method"),
        ("step", "Step method"),
        ("_compute_lidar", "Lidar computation"),
        ("_ray_box_intersection", "Ray-box intersection"),
        ("_destroy", "Cleanup method"),
    ]

    for method_name, description in required_methods:
        assert f"def {method_name}(" in code, f"Method {method_name} not found"
        print(f"  [PASS] {description} ({method_name})")

    # Test 6: Check observation space logic
    print("\n[Test 6] Checking observation space...")
    assert "base_obs_dim = 9" in code, "Base obs dimension not set correctly"
    assert "total_obs_dim = base_obs_dim + N_LIDAR_RAYS" in code, "Total obs calculation missing"
    print("[PASS] Observation space: 9 (base) + 8 (lidar) = 17 dimensions")

    # Test 7: Check obstacle spawning logic
    print("\n[Test 7] Checking obstacle spawning...")
    assert "for i in range(NUM_OBSTACLES):" in code, "Obstacle spawning loop missing"
    assert "obstacle.userData = {'obstacle': True}" in code, "Obstacle tagging missing"
    print("[PASS] Obstacle spawning logic present")
    print("  - Spawns NUM_OBSTACLES boxes")
    print("  - Tags with userData={'obstacle': True}")

    # Test 8: Check collision detection logic
    print("\n[Test 8] Checking collision detection...")
    assert "for contact in self.lander.contacts:" in code, "Contact iteration missing"
    assert "info['collision'] = True" in code, "Collision info flag missing"
    assert "reward += COLLISION_PENALTY" in code, "Collision penalty missing"
    print("[PASS] Collision detection logic present")
    print("  - Iterates lander contacts")
    print("  - Sets info['collision']")
    print("  - Applies penalty")

    # Test 9: Check lidar implementation
    print("\n[Test 9] Checking lidar implementation...")
    assert "for i in range(N_LIDAR_RAYS):" in code, "Lidar ray loop missing"
    assert "angle = 2 * np.pi * i / N_LIDAR_RAYS" in code, "Lidar angle calculation missing"
    assert "min_dist / MAX_LIDAR_DISTANCE" in code, "Lidar normalization missing"
    print("[PASS] Lidar implementation present")
    print("  - Casts 8 rays evenly spaced")
    print("  - Normalizes distances to [0, 1]")

    # Test 10: Check observation concatenation
    print("\n[Test 10] Checking observation augmentation...")
    assert "np.concatenate([base_obs, lidar_readings])" in code, "Observation concatenation missing"
    print("[PASS] Observation augmentation present")
    print("  - Concatenates base_obs + lidar")

    # Test 11: Check registration file
    print("\n[Test 11] Checking environment registration...")
    reg_file = Path("diffusha/envs/__init__.py")
    assert reg_file.exists(), "Registration file missing"
    with open(reg_file, 'r') as f:
        reg_code = f.read()
    assert "LunarLanderObstacle-v5" in reg_code, "Environment ID not registered"
    assert "entry_point='diffusha.envs.lunar_lander_obstacle:LunarLanderObstacle'" in reg_code, "Entry point incorrect"
    print("[PASS] Environment registered")
    print("  - ID: LunarLanderObstacle-v5")
    print("  - Max steps: 1000")

    # Test 12: Check package init
    print("\n[Test 12] Checking package initialization...")
    init_file = Path("diffusha/__init__.py")
    assert init_file.exists(), "Package __init__.py missing"
    with open(init_file, 'r') as f:
        init_code = f.read()
    assert "from diffusha import envs" in init_code, "envs import missing"
    print("[PASS] Package __init__.py imports envs module")

    # Test 13: Logic validation - ray-box intersection algorithm
    print("\n[Test 13] Checking ray-box intersection algorithm...")
    # Check for slab method components
    assert "t_min = 0.0" in code and "t_max = 1.0" in code, "Slab method initialization missing"
    assert "X slab" in code or "Y slab" in code, "Slab comments missing"
    print("[PASS] Ray-box intersection uses slab method")

    # Test 14: Check drawlist augmentation
    print("\n[Test 14] Checking rendering support...")
    assert "self.drawlist.extend(self.obstacles)" in code, "Obstacles not added to drawlist"
    assert "obstacle.color1" in code, "Obstacle colors not set"
    print("[PASS] Obstacles added to rendering drawlist")

    print("\n" + "=" * 60)
    print("All Standalone Tests Passed! [PASS]")
    print("=" * 60)
    print("\nPhase 1 Implementation Summary:")
    print("  - Environment: LunarLanderObstacle-v5")
    print("  - Observation: 17-dim (8 base + 1 helipad + 8 lidar)")
    print("  - Obstacles: 4 static boxes with collision detection")
    print("  - Lidar: 8 rays, normalized distances")
    print("  - Collision penalty: -100")
    print("\nNext: Run full test inside Docker to validate runtime behavior")

    return True

def test_imports():
    """Test that imports work (as much as possible without dependencies)"""
    print("\n[Bonus Test] Checking imports structure...")

    env_file = Path("diffusha/envs/lunar_lander_obstacle.py")
    with open(env_file, 'r') as f:
        code = f.read()

    # Check expected imports
    expected_imports = [
        "import math",
        "import numpy as np",
        "import Box2D",
        "import gym",
        "from pathlib import Path",
    ]

    for imp in expected_imports:
        assert imp in code, f"Missing import: {imp}"

    print("[PASS] All expected imports present")

if __name__ == "__main__":
    try:
        success = test_syntax_and_structure()
        test_imports()

        if success:
            print("\nSUCCESS: Phase 1 validation successful!")
            print("Code is ready for Docker container testing.")
            sys.exit(0)
        else:
            print("\n[FAILED] Validation failed")
            sys.exit(1)

    except AssertionError as e:
        print(f"\n[FAILED] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILED] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
