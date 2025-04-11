import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import pulp  # For linear programming
from sklearn.linear_model import LinearRegression
import joblib  # For saving/loading models

logger = logging.getLogger(__name__)

class ILPAction(Enum):
    """Actions that the ILP optimizer can take"""
    USE_SMALL_MODEL_1 = 0 # neural_model
    USE_SMALL_MODEL_2 = 1 # ngram
    USE_SMALL_MODEL_3 = 2
    DISABLE_SPEC_DECODING = 3

class ILPOptimizer:
    """An ILP-based optimizer for dynamic model switching.
    
    This optimizer uses integer linear programming to select the optimal
    speculative sampling method (SSM) from M choices including M-1 small models
    and the option to disable speculative decoding.
    
    The optimization problem is formulated as:
    
    max_x  g_x * x
    s.t.   sum_i x_i = 1, for all i in M
           x_i in {0, 1}, for all i in M
           sum_i v_i <= Mem
           sum_i g_i * x_i >= g_0
    
    where:
    - x is a one-hot vector representing the chosen SSM
    - g_x is the throughput under selection x
    - v_i is the memory usage of each model
    - Mem is the total available memory
    - g_0 is the minimum acceptable throughput
    """
    
    def __init__(self, 
                 llm_engine,
                 num_small_models: int = 3,
                 reward_window_size: int = 10,
                 min_samples_per_model: int = 3,
                 cooldown_period: float = 30.0,  # seconds
                 min_acceptable_throughput: float = 10.0):  # tokens/s
        self.llm_engine = llm_engine
        # Create actions list based on number of small models
        self.actions = []
        for i in range(num_small_models):
            if i == 0:
                self.actions.append(ILPAction.USE_SMALL_MODEL_1)
            elif i == 1:
                self.actions.append(ILPAction.USE_SMALL_MODEL_2)
            elif i == 2:
                self.actions.append(ILPAction.USE_SMALL_MODEL_3)
        self.actions.append(ILPAction.DISABLE_SPEC_DECODING)
        
        # Initialize counts for each model
        self.counts = {action: 0 for action in self.actions}
        
        # Initialize throughput history for each model
        self.throughput_history = {action: [] for action in self.actions}
        
        # Running metrics
        self.request_load_history = []
        self.memory_usage_history = []
        self.acceptance_rate_history = []
        self.spec_length_history = []
        
        # Linear models for T_D and T_V prediction using sklearn
        self.t_d_model = LinearRegression()
        self.t_v_model = LinearRegression()
        self.t_d_model_trained = False
        self.t_v_model_trained = False
        
        # Memory usage for each model
        self.model_memory_usage = {action: 0.0 for action in self.actions}
        
        # State tracking
        self.last_action = None
        self.last_action_time = 0.0
        self.current_state = None
        
        # Configuration
        self.reward_window_size = reward_window_size # history window size
        self.min_samples_per_model = min_samples_per_model # minimum samples per model
        self.cooldown_period = cooldown_period # cooldown period
        self.min_acceptable_throughput = min_acceptable_throughput # minimum acceptable throughput
        
        # Current model state
        self.current_model_index = -1  # -1 means no speculative decoding
        
    def record_metrics(self, 
                      throughput: float, 
                      acceptance_rate: Optional[float] = None,
                      spec_length: Optional[int] = None) -> None:
        """Record current system metrics"""
        # Record main metrics
        if self.last_action is not None:
            self.throughput_history[self.last_action].append(throughput)
            # Keep history bounded
            if len(self.throughput_history[self.last_action]) > self.reward_window_size:
                self.throughput_history[self.last_action].pop(0)
        
        # Record speculative decoding metrics if available
        # Record speculative decoding metrics if available
        if acceptance_rate is not None:
            self.acceptance_rate_history.append(acceptance_rate)
        if spec_length is not None:
            self.spec_length_history.append(spec_length)
        avg_acceptance_rate = sum(self.acceptance_rate_history)/len(self.acceptance_rate_history)
        # Keep history bounded
        if len(self.request_load_history) > self.reward_window_size:
            self.request_load_history.pop(0)
            self.throughput_history.pop(0)
            if self.acceptance_rate_history:
                self.acceptance_rate_history.pop(0)
            
            if self.spec_length_history:
                self.spec_length_history.pop(0)
        
        # Store current state for calculations
        self.current_state = {
            "throughput": throughput,
            "acceptance_rate": avg_acceptance_rate,
            "spec_length": spec_length if spec_length is not None else 0
        }
        
    def update_model_memory_usage(self, action: ILPAction, memory_usage: float) -> None:
        """Update memory usage for a specific model"""
        self.model_memory_usage[action] = memory_usage
    
    def _train_linear_models(self, perf_data_path: str, save_path: str) -> None:
        if self.t_d_model_trained and self.t_v_model_trained:
            return
        from simple_spec.perf_model import PrefillPerformanceModel
        model = PrefillPerformanceModel()
        model.train(perf_data_path,  save_path)
        return model
    
    def load_model(self, model_path_d: str, model_path_v: str) -> None:
        if self.t_d_model_trained and self.t_v_model_trained:
            return
        self.t_d_model = joblib.load(model_path_d)
        self.t_v_model = joblib.load(model_path_v)
        self.t_d_model_trained = True
        self.t_v_model_trained = True
    
    def _predict_throughput(self, 
                           action: ILPAction, 
                           acceptance_rate: float,
                           spec_length: int) -> float:
        """Predict throughput for a specific action using the formula
        
        g_x = (B * (1 - α^(γ+1))/(1 - α)) / 
              (γ * T_D(B, ∑S_b, 1) + T_V(B, ∑S_b, γ) + C_swap)
        """
        scheduler_outputs = self.llm_engine.try_scheduler()
        request_load = len(scheduler_outputs.scheduled_seq_groups) - scheduler_outputs.num_prefill_groups
        context_tokens = scheduler_outputs.num_cached_tokens
        batch_tokens = scheduler_outputs.num_batched_tokens 
        # If we have historical data for this action, use the average as baseline
        # if self.throughput_history[action] and len(self.throughput_history[action]) >= self.min_samples_per_model:
        #     return np.mean(self.throughput_history[action])
        
        # Otherwise, predict using the formula
        # B is the batch size (number of requests)
        B = max(request_load, 1)
        
        # Get current memory usage
        memory_usage = 0.0
        if self.current_state:
            memory_usage = self.current_state["memory_usage"]
        
        # α is the acceptance rate
        alpha = acceptance_rate if acceptance_rate > 0 else 0.5  # default if unknown
        
        # γ is the speculation length
        gamma = spec_length if spec_length > 0 else 3  # default if unknown
        if action == ILPAction.DISABLE_SPEC_DECODING:
            gamma = 1
        # Calculate numerator: number of tokens processed
        # B * (1 - α^(γ+1))/(1 - α)
        if abs(1 - alpha) < 1e-6:  # alpha is close to 1
            tokens_processed = B * (gamma + 1)
        else:
            tokens_processed = B * (1 - alpha**(gamma+1)) / (1 - alpha)
        
        # Predict T_D (draft model time) using sklearn model
        if self.t_d_model_trained and action != ILPAction.DISABLE_SPEC_DECODING:
            # Use sklearn's predict method for better accuracy
            feature_vector = np.array([context_tokens,B])
            t_d =   float(self.t_d_model.predict(feature_vector)[0])
        else:
            # Default model if not trained
            t_d = 0.01 * B
        
        # Predict T_V (verification time) using sklearn model
        if self.t_v_model_trained:
            # Using sklearn's predict method
            feature_vector = np.array([context_tokens,B*gamma])
            t_v = float(self.t_v_model.predict(feature_vector)[0])
        else:
            # Default model if not trained
            t_v = 0.005 * B
        
        # Calculate C_swap (cost of switching models)
        # If we're already using this model, no swap cost
        # Otherwise, assume some fixed cost
        if self.current_model_index == action.value:
            c_swap = 0.0
        else:
            c_swap = 1  # placeholder value
        
        # Calculate denominator: total time
        # γ * T_D(B, ∑S_b, 1) + T_V(B, ∑S_b, γ) + C_swap
        total_time = gamma * t_d + t_v + c_swap
        
        # Avoid division by zero
        if total_time <= 0:
            return 0.0
        
        # Calculate throughput: tokens / time
        throughput = tokens_processed / total_time
        
        return throughput
    
    def _has_min_samples(self) -> bool:
        """Check if all models have been tried the minimum number of times"""
        return all(self.counts[action] >= self.min_samples_per_model for action in self.actions)
    
    def solve_ilp(self) -> Optional[ILPAction]:
        """Solve the ILP optimization problem to select the best model"""
        if self.current_state is None:
            return None
        
        # Create the ILP problem
        problem = pulp.LpProblem("ModelSelection", pulp.LpMaximize)
        
        # Create decision variables (binary x_i for each model)
        x = {action: pulp.LpVariable(f"x_{action.value}", cat=pulp.LpBinary) 
             for action in self.actions}
        
        # Get current metrics
        acceptance_rate = self.current_state["acceptance_rate"]
        spec_length = self.current_state["spec_length"]
        
        # Predict throughput for each model
        throughput = {action: self._predict_throughput( action, acceptance_rate, spec_length) 
                     for action in self.actions}
        self.min_acceptable_throughput = throughput[ILPAction.DISABLE_SPEC_DECODING]
        # Objective function: maximize throughput
        problem += pulp.lpSum([throughput[action] * x[action] for action in self.actions])
        
        # Constraint 1: select exactly one model
        problem += pulp.lpSum([x[action] for action in self.actions]) == 1
        
        # Constraint 2: memory usage constraint
        total_memory = pulp.lpSum([self.model_memory_usage[action] * x[action] for action in self.actions])
        available_memory = 1.0  # Normalized to 1.0
        problem += total_memory <= available_memory
        
        # Constraint 3: minimum throughput requirement
        problem += pulp.lpSum([throughput[action] * x[action] for action in self.actions]) >= self.min_acceptable_throughput
        
        # Solve the problem
        problem.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Check if a solution was found
        if problem.status != pulp.LpStatusOptimal:
            logger.warning("No optimal solution found in ILP")
            return None
        
        # Get the selected model
        selected_action = None
        for action in self.actions:
            if pulp.value(x[action]) > 0.5:  # Binary variable is approximately 1
                selected_action = action
                break
        
        # Update counts and last action time
        if selected_action is not None:
            self.counts[selected_action] += 1
            self.last_action_time = time.time()
            self.last_action = selected_action
        
        return selected_action
    
    def select_action(self, 
                     throughput: float, 
                     acceptance_rate: Optional[float] = None,
                     spec_length: Optional[int] = None) -> Optional[ILPAction]:
        """Select the best action based on current metrics"""
        # Record current metrics
        self.record_metrics(throughput, 
                           acceptance_rate, spec_length)
        
        # Check if we're in cooldown period
        current_time = time.time()
        if self.last_action_time > 0 and current_time - self.last_action_time < self.cooldown_period:
            logger.info("In cooldown period, no action selected")
            return None
        
        # Train linear models if needed
        self._train_linear_models()
        self.load_model()
        
        # Solve the ILP problem
        action = self.solve_ilp()
        
        # Check if we should actually switch
        if action is not None and action.value == self.current_model_index:
            # We're already using this model, so no need to switch
            return None
        
        logger.info(f"ILP optimizer selected action: {action}")
        return action
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the optimizer"""
        if self.current_state is None:
            return {"status": "not_initialized"}
        
        avg_throughput = {}
        for action in self.actions:
            if self.throughput_history[action]:
                avg_throughput[action.name] = np.mean(self.throughput_history[action])
            else:
                avg_throughput[action.name] = 0.0
        
        return {
            "current_model_index": self.current_model_index,
            "last_action": self.last_action.name if self.last_action else None,
            "counts": {action.name: count for action, count in self.counts.items()},
            "average_throughput": avg_throughput,
            "current_metrics": self.current_state,
            "model_memory_usage": {action.name: usage for action, usage in self.model_memory_usage.items()},
            "t_d_model_trained": self.t_d_model_trained,
            "t_v_model_trained": self.t_v_model_trained
        }
        
    def save_models(self, t_d_model_path: str, t_v_model_path: str) -> bool:
        """Save trained linear regression models to files
        
        Args:
            t_d_model_path: Path to save the draft model
            t_v_model_path: Path to save the verification model
            
        Returns:
            bool: Whether the models were successfully saved
        """
        if not self.t_d_model_trained or not self.t_v_model_trained:
            logger.warning("Models are not trained yet, cannot save")
            return False
            
        try:
            # Save the sklearn models
            joblib.dump(self.t_d_model, t_d_model_path)
            joblib.dump(self.t_v_model, t_v_model_path)
            logger.info(f"Models saved to {t_d_model_path} and {t_v_model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False
            
    def load_models(self, t_d_model_path: str, t_v_model_path: str) -> bool:
        """Load trained linear regression models from files
        
        Args:
            t_d_model_path: Path to load the draft model from
            t_v_model_path: Path to load the verification model from
            
        Returns:
            bool: Whether the models were successfully loaded
        """
        try:
            # Load the sklearn models
            self.t_d_model = joblib.load(t_d_model_path)
            self.t_v_model = joblib.load(t_v_model_path)
            
            # Mark models as trained
            self.t_d_model_trained = True
            self.t_v_model_trained = True
            
            logger.info(f"Models loaded from {t_d_model_path} and {t_v_model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def import_performance_data(self, 
                           perf_data_path: str, 
                           train_models: bool = True) -> bool:
        """Import performance measurements from a JSON file for model training
        
        Args:
            perf_data_path: Path to performance data in JSON format
            train_models: Whether to immediately train models after import
            
        Returns:
            bool: Whether the data was successfully imported
        """
        try:
            import json
            
            # Load the performance data
            with open(perf_data_path, 'r') as f:
                perf_data = json.load(f)
                
            # Reset existing data
            self.request_load_history = []
            self.memory_usage_history = []
            
            t_d_data = []  # Draft model time data
            t_v_data = []  # Verification model time data
            
            # Process each entry in the performance data
            for entry in perf_data:
                # Extract metrics
                if "batch_size" in entry:
                    self.request_load_history.append(entry["batch_size"])
                if "memory_usage" in entry:
                    self.memory_usage_history.append(entry["memory_usage"])
                elif "avg_memory_usage" in entry:  # Alternative field name
                    self.memory_usage_history.append(entry["avg_memory_usage"])
                else:
                    # If memory usage is not available, use a placeholder
                    self.memory_usage_history.append(0.5)  # Normalized placeholder
                    
                # Extract draft and verification times if available
                if "draft_time" in entry:
                    t_d_data.append(entry["draft_time"])
                if "verification_time" in entry:
                    t_v_data.append(entry["verification_time"])
            
            # If we have real timing data, override the placeholders
            if t_d_data and len(t_d_data) == len(self.request_load_history):
                # Create real Y values for draft time model
                self.t_d_real_data = t_d_data
                logger.info(f"Imported {len(t_d_data)} real measurements for draft time")
            
            if t_v_data and len(t_v_data) == len(self.request_load_history):
                # Create real Y values for verification time model
                self.t_v_real_data = t_v_data
                logger.info(f"Imported {len(t_v_data)} real measurements for verification time")
            
            # Train the models if requested
            if train_models:
                self._train_models_with_real_data()
                
            logger.info(f"Successfully imported performance data from {perf_data_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import performance data: {e}")
            return False
    
    def _train_models_with_real_data(self) -> None:
        """Train models using real measurement data if available"""
        if len(self.request_load_history) < self.min_samples_per_model:
            logger.warning("Not enough data points to train models")
            return
            
        # Prepare features: using both request_load and memory_usage
        X = np.array([
            [req, mem] 
            for req, mem in zip(self.request_load_history, self.memory_usage_history)
        ])
        
        # Train T_D model (draft model time) if we have real data
        if hasattr(self, 't_d_real_data') and len(self.t_d_real_data) == len(self.request_load_history):
            Y_d = np.array(self.t_d_real_data)
            
            # Train the model using sklearn's LinearRegression
            if len(X) > 1:
                self.t_d_model.fit(X, Y_d)
                self.t_d_model_trained = True
                logger.info(f"T_D model trained with real data, R² score: {self.t_d_model.score(X, Y_d):.3f}")
                logger.debug(f"T_D model coefficients: {self.t_d_model.coef_}")
        
        # Train T_V model (verification time) if we have real data
        if hasattr(self, 't_v_real_data') and len(self.t_v_real_data) == len(self.request_load_history):
            Y_v = np.array(self.t_v_real_data)
            
            # Train the model using sklearn's LinearRegression
            if len(X) > 1:
                self.t_v_model.fit(X, Y_v)
                self.t_v_model_trained = True
                logger.info(f"T_V model trained with real data, R² score: {self.t_v_model.score(X, Y_v):.3f}")
                logger.debug(f"T_V model coefficients: {self.t_v_model.coef_}") 
    
    def visualize_models(self, 
                        save_path: Optional[str] = None, 
                        show_plot: bool = False) -> bool:
        """Visualize the trained models and their predictions
        
        Args:
            save_path: Path to save the visualization (None to not save)
            show_plot: Whether to display the plot (requires matplotlib in interactive mode)
            
        Returns:
            bool: Whether the visualization was successful
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Check if models are trained
            if not self.t_d_model_trained and not self.t_v_model_trained:
                logger.warning("No trained models to visualize")
                return False
            
            # Create a figure with two 3D subplots
            fig = plt.figure(figsize=(15, 7))
            
            # Define color maps
            cmap_d = plt.cm.viridis
            cmap_v = plt.cm.plasma
            
            # If we have training data, visualize it
            if len(self.request_load_history) > 0:
                # Prepare data
                X = np.array([
                    [req, mem] 
                    for req, mem in zip(self.request_load_history, self.memory_usage_history)
                ])
                
                # Define grid for predictions
                x_range = np.linspace(
                    max(1, min(self.request_load_history) * 0.8),
                    max(self.request_load_history) * 1.2,
                    20
                )
                y_range = np.linspace(
                    max(0.1, min(self.memory_usage_history) * 0.8),
                    max(self.memory_usage_history) * 1.2,
                    20
                )
                X_grid, Y_grid = np.meshgrid(x_range, y_range)
                grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
                
                # Plot T_D model if trained
                if self.t_d_model_trained:
                    ax_d = fig.add_subplot(1, 2, 1, projection='3d')
                    ax_d.set_title('Draft Model (T_D) Predictions')
                    ax_d.set_xlabel('Request Load')
                    ax_d.set_ylabel('Memory Usage')
                    ax_d.set_zlabel('T_D (Time)')
                    
                    # Predict on grid
                    Z_pred_d = self.t_d_model.predict(grid_points).reshape(X_grid.shape)
                    
                    # Plot surface
                    surf_d = ax_d.plot_surface(
                        X_grid, Y_grid, Z_pred_d, 
                        cmap=cmap_d, alpha=0.7, antialiased=True
                    )
                    fig.colorbar(surf_d, ax=ax_d, shrink=0.5, aspect=5)
                    
                    # Plot actual data points if we have real data
                    if hasattr(self, 't_d_real_data'):
                        Y_d = np.array(self.t_d_real_data)
                        ax_d.scatter(
                            X[:, 0], X[:, 1], Y_d, 
                            c='r', marker='o', s=50, alpha=1.0, 
                            label='Actual Measurements'
                        )
                    
                    # Add R² score to title if we have real data
                    if hasattr(self, 't_d_real_data'):
                        Y_d = np.array(self.t_d_real_data)
                        ax_d.set_title(
                            f'Draft Model (T_D) Predictions\nR² = {self.t_d_model.score(X, Y_d):.3f}'
                        )
                
                # Plot T_V model if trained
                if self.t_v_model_trained:
                    ax_v = fig.add_subplot(1, 2, 2, projection='3d')
                    ax_v.set_title('Verification Model (T_V) Predictions')
                    ax_v.set_xlabel('Request Load')
                    ax_v.set_ylabel('Memory Usage')
                    ax_v.set_zlabel('T_V (Time)')
                    
                    # Predict on grid
                    Z_pred_v = self.t_v_model.predict(grid_points).reshape(X_grid.shape)
                    
                    # Plot surface
                    surf_v = ax_v.plot_surface(
                        X_grid, Y_grid, Z_pred_v, 
                        cmap=cmap_v, alpha=0.7, antialiased=True
                    )
                    fig.colorbar(surf_v, ax=ax_v, shrink=0.5, aspect=5)
                    
                    # Plot actual data points if we have real data
                    if hasattr(self, 't_v_real_data'):
                        Y_v = np.array(self.t_v_real_data)
                        ax_v.scatter(
                            X[:, 0], X[:, 1], Y_v, 
                            c='r', marker='o', s=50, alpha=1.0,
                            label='Actual Measurements'
                        )
                    
                    # Add R² score to title if we have real data
                    if hasattr(self, 't_v_real_data'):
                        Y_v = np.array(self.t_v_real_data)
                        ax_v.set_title(
                            f'Verification Model (T_V) Predictions\nR² = {self.t_v_model.score(X, Y_v):.3f}'
                        )
            
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Model visualization saved to {save_path}")
            
            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
                
            return True
            
        except ImportError as e:
            logger.error(f"Visualization requires matplotlib: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to visualize models: {e}")
            return False 