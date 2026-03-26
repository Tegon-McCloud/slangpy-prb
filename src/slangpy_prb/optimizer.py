import slangpy as spy

from . import SceneVariables

class Optimizer:

    def __init__(self):
        super().__init__()

    def step(self, command_encoder: spy.CommandEncoder): ...

class GradientDescent(Optimizer):
    def __init__(
        self,
        device: spy.Device,
        variables: SceneVariables,
        gradient: SceneVariables,
        learning_rate: float,
    ):
        super().__init__()

        self.device = device
        self.variables = variables
        self.gradient = gradient

        self.learning_rate = learning_rate
        self.program = device.load_program("shaders/gradient_descent.slang", ["main"])
        self.kernel = device.create_compute_kernel(self.program)

    def step(self, command_encoder: spy.CommandEncoder):
        self.kernel.dispatch(
            thread_count=[self.variables.shape.num_parameters, 1, 1],
            vars={
                "gradient_descent": {
                    "learning_rate": self.learning_rate,
                },
                "variables": self.variables.parameter_buffer,
                "gradient": self.gradient.parameter_buffer,
            },
            command_encoder=command_encoder,
        )
        
class Adam(Optimizer):
    def __init__(
        self,
        device: spy.Device,
        variables: SceneVariables,
        gradient: SceneVariables,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__()

        self.device = device
        self.variables = variables
        self.gradient = gradient

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step_counter = 0
        self.moment1 = SceneVariables(self.device, self.variables.shape)
        self.moment2 = SceneVariables(self.device, self.variables.shape)

        self.program = device.load_program("shaders/adam.slang", ["main"])
        self.kernel = device.create_compute_kernel(self.program)

    def step(self, command_encoder: spy.CommandEncoder):
        self.step_counter += 1

        self.kernel.dispatch(
            thread_count=[self.variables.shape.num_parameters, 1, 1],
            vars={
                "adam": {
                    "learning_rate": self.learning_rate,
                    "beta1": self.beta1,
                    "scale1": 1.0 / (1.0 - self.beta1 ** self.step_counter),
                    "beta2": self.beta2,
                    "scale2": 1.0 / (1.0 - self.beta2 ** self.step_counter), 
                    "epsilon": self.epsilon,
                },
                "variables": self.variables.parameter_buffer,
                "gradient": self.gradient.parameter_buffer,
            },
            command_encoder=command_encoder,
        )
        
        


