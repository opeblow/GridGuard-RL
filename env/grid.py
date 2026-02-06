class Grid:
    def __init__(self):
        self.nominal_frequency=50.0
        self.inertia=5.0
        self.damping=1.0
        self.dt=0.05

        self.reset()

    def reset(self):
        self.frequency=self.nominal_frequency
        self.load=1000
        self.generation=1000
        return self.frequency
    
    def step(self):
        power_imbalance=self.generation - self.load
        df_dt= (power_imbalance/self.inertia - self.damping * (self.frequency -self.nominal_frequency))
        self.previous_frequency=self.frequency
        self.frequency+=df_dt * self.dt
        self.frequency=max(45.0,min(55.0,self.frequency))
        rocof=(self.frequency - self.previous_frequency) / self.dt
        return self.frequency,rocof
    
    def change_load(self,delta):
        self.load +=delta
        if self.load < 0:
            self.load=0

    def change_generation(self,delta):
        self.generation+=delta
        if self.generation < 0:
            self.generation =0

    def check_threshold(self):
        if self.frequency < 47.5:
            print(f"Alert:Frequency at {self.frequency:.2f}Hz -Collapse Threshold Breached")
            return True
        return False
    
    def perform_load_shedding(self,amount):
        print(f"Emergency:Shedding {amount} units of load to stabilize grid")
        self.load-=amount

    def recover_frequency(self,target=50.0):
        increment=10
        print(f"Recovery:Increasing Generationby {increment} units")
        self.generation +=increment
