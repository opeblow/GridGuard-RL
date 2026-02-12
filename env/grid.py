class Grid:
    def __init__(self):
        self.nominal_frequency=50.0
        self.inertia=5.0
        self.damping=1.0
        self.dt=0.05

        self.max_gen_change = 200.0

        self.reset()

    def reset(self):
        self.frequency=self.nominal_frequency
        self.load=1000
        self.generation=1000
        self._collapse_alerted = False
        self.violation_time = 0
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
        if self.frequency < 47.5:
            self.violation_time +=1
            if self.violation_time ==1:
                print(f"Alert:Frequency at {self.frequency:.2f}Hz- collapse Threshold breached")
                self._collapse_alerted = True
            if self.violation_time > 5:
                print(f"Blackout:Frequency remained below the threshold for  {self.violation_time}steps")
                return True
            return False
        else:
            if self.violation_time > 0:
                print(f"Recovery:Frequency restorefd to {self.frequency:.2f}Hz (was violated for {self.violation_time})")
                self._collapse_alerted = False
            self.violation_time = 0
            return False              
    
    def perform_load_shedding(self,amount):
        print(f"Emergency:Shedding {amount} units of load to stabilize grid")
        self.load-=amount

    def recover_frequency(self,target=50.0):
        increment=10
        print(f"Recovery:Increasing Generationby {increment} units")
        self.generation +=increment


     
     