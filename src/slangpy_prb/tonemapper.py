import slangpy as spy

class Tonemapper:
    def __init__(
        self,
        device: spy.Device,
    ):
        self.device = device
        self.program = self.device.load_program("tonemap.slang", ["main"])

        self.kernel = self.device.create_compute_kernel(self.program)

    def tonemap(
        self,
        command_encoder: spy.CommandEncoder,
        input: spy.Texture,
        output: spy.Texture,    
    ):
        self.kernel.dispatch(
            thread_count=[output.width, output.height, 1],
            vars={
                "tonemapper": {
                    "input": input,
                    "output": output,
                },
            },
            command_encoder=command_encoder,
        )

