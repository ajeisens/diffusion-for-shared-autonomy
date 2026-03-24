#!/usr/bin/env python3
"""Minimal pygame test to verify display connection"""

import pygame
import sys

print("Python version:", sys.version)
print("Pygame version:", pygame.version.ver)

print("Initializing pygame...")
result = pygame.init()
print(f"Pygame init result: {result}")

print("Available video drivers:", pygame.display.get_driver())

try:
    print("Creating display...")
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Pygame Test")
    print("Display created successfully!")

    # Draw something simple
    screen.fill((0, 100, 200))
    pygame.display.flip()
    print("Screen updated")

    # Wait a bit
    import time
    print("Waiting 5 seconds...")
    time.sleep(5)

    print("Test completed successfully!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    pygame.quit()
    print("Pygame quit")
