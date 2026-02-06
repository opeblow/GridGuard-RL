from env.grid import Grid

def test_stable_frequency():
    grid=Grid()
    print("Starting Test 1: Stable Frequency Test\n")
    steps=100
    for step in range (steps):
        frequency,rocof =grid.step()
        if step % 10 == 0:
            print(f"Step {step:03d} |" f"Frequency: {frequency:.4f}HZ |" f"RoCoF:{rocof:.4f} HZ/S")
    if abs(frequency - grid.nominal_frequency) < 0.05:
        print("\n PASS:Frequency remained stable at normal value")
    else:
        print("\n FAIL:Frequency drifted unexpectedly")
if __name__ == "__main__":
    test_stable_frequency()
            