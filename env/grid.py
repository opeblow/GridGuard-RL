class Grid:
    def __init__(self):
        self.nominal_frequency=50.0
        self.inertia=5.0
        self.damping=1.0
        self.dt=0.05

        # Maximum generation change (MW) applied per environment step
        self.max_gen_change = 50.0

        self.reset()

    def reset(self):
        self.frequency=self.nominal_frequency
        self.load=1000
        self.generation=1000
        self._collapse_alerted = False
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
        # Apply generation change with a maximum ramp rate to make control smoother
        # and to give the agent time to react. Clip delta per timestep.
        if delta is None:
            return
        applied = delta
        if applied > self.max_gen_change:
            applied = self.max_gen_change
        elif applied < -self.max_gen_change:
            applied = -self.max_gen_change

        self.generation += applied
        if self.generation < 0:
            self.generation = 0

    def check_threshold(self):
        # Return True when frequency falls below a safe threshold.
        # Only emit a single alert per episode (prevents log spam).
        if self.frequency < 47.5:
            if not getattr(self, '_collapse_alerted', False):
                print(f"Alert:Frequency at {self.frequency:.2f}Hz -Collapse Threshold Breached")
                self._collapse_alerted = True
            return True
        return False
    
    def perform_load_shedding(self,amount):
        print(f"Emergency:Shedding {amount} units of load to stabilize grid")
        self.load-=amount

    def recover_frequency(self,target=50.0):
        increment=10
        print(f"Recovery:Increasing Generationby {increment} units")
        self.generation +=increment


     
     