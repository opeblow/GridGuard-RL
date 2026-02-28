from env.grid import Grid
grid=Grid()
print("Starting Test 4...")
print("Generator Failure:Loss of 150 units")
grid.change_generation(-150)
for i in range(10):
    freq,rocof=grid.step()
    print(f"Step {i} | Frequency:{freq:.4f}|RocoF:{rocof:.4f}")
