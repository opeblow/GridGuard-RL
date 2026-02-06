from env.grid import Grid
grid=Grid()
print("Starting Test 3..")
for _ in range(10):
    grid.step()
grid.change_load(200)
for i in range(20):
    freq,rocof=grid.step()
    print(f"Step {i:02d} |Frequency:{freq:.4f}Hz | RoCoF:{rocof:.4f}Hz/s")
    