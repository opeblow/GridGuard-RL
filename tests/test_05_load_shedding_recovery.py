from env.grid import Grid
grid=Grid()
print("Starting Test 5")
grid.change_generation(-200)
for step in range(10):
    freq,rocof=grid.step()
    print(f"Step:{step} |Freq:{freq:.4f}Hz |RocoF:.4f")

    if grid.check_threshold() and step ==3:
        grid.perform_load_shedding(200)

print("\nStarting Recovery...")
while grid.frequency < 49.99:
    grid.recover_frequency()
    grid.step()
    print(f"Recovery |Frequency : {grid.frequency:.4f}Hz")

print("Grid Restored To Normal Frequency")
