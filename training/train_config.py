"""
Training configurations for GridGuard RL Agent
Provides FAST_MODE for rapid iteration (2-5 min) and FULL_MODE for production (30+ min)
"""

class FastModeConfig:
  
    
   
    TOTAL_TIMESTEPS = 50_000           
    EVAL_FREQ = 5_000                 
    SAVE_FREQ = 10_000               
    N_EVAL_EPISODES = 3                
    FINAL_EVAL_EPISODES = 5           
    
    SKIP_BASELINE_EVAL = True           
    SKIP_INTERMEDIATE_LOGGING = True   
    
    # PPO HYPERPARAMETERS
    PPO_PARAMS = {
        'learning_rate': 3e-4,          
        'n_steps': 512,               
        'batch_size': 64,              
        'n_epochs': 3,                 
        'gamma': 0.99,                 
        'gae_lambda': 0.95,             
        'clip_range': 0.2,             
        'ent_coef': 0.01,               
        'vf_coef': 0.5,                 
        'max_grad_norm': 0.5,         
        'policy_kwargs': {
            'net_arch': [32, 32]       
        }
    }
    
    # EARLY STOPPING CRITERIA
    EARLY_STOPPING_ENABLED = True
    TARGET_BLACKOUT_RATE = 0.05         
    TARGET_MEAN_FREQ_DEV = 0.5         
    EARLY_STOPPING_PATIENCE = 3        
    
    # DESCRIPTION
    MODE_NAME = "FAST"
    EXPECTED_DURATION = "2-5 minutes"
    USE_CASE = "Rapid iteration, local testing, portfolio demo"


class FullModeConfig:
   
    
    # TRAINING PARAMETERS
    TOTAL_TIMESTEPS = 500_000           
    EVAL_FREQ = 10_000                
    SAVE_FREQ = 50_000              
    N_EVAL_EPISODES = 10              
    FINAL_EVAL_EPISODES = 20          
    
    SKIP_BASELINE_EVAL = False          
    SKIP_INTERMEDIATE_LOGGING = False   
    
    # PPO HYPERPARAMETERS
    PPO_PARAMS = {
        'learning_rate': 3e-4,          
        'n_steps': 2048,                
        'batch_size': 64,               
        'n_epochs': 10,                 
        'gamma': 0.99,                 
        'gae_lambda': 0.95,             
        'clip_range': 0.2,              
        'ent_coef': 0.01,               
        'vf_coef': 0.5,               
        'max_grad_norm': 0.5,           
        'policy_kwargs': {
            'net_arch': [64, 64]        
        }
    }
    
    # EARLY STOPPING CRITERIA
    EARLY_STOPPING_ENABLED = True
    TARGET_BLACKOUT_RATE = 0.05         
    TARGET_MEAN_FREQ_DEV = 0.3          
    EARLY_STOPPING_PATIENCE = 5        
    
    # DESCRIPTION
    MODE_NAME = "FULL"
    EXPECTED_DURATION = "30-60 minutes"
    USE_CASE = "Production deployment, comprehensive testing, final evaluation"


class TrainingConfig:
    """Base configuration selector"""
    
    FAST = FastModeConfig
    FULL = FullModeConfig
    
    @staticmethod
    def get_config(mode: str = "fast"):
        
        if mode.lower() == "fast":
            return TrainingConfig.FAST
        elif mode.lower() == "full":
            return TrainingConfig.FULL
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'fast' or 'full'")


