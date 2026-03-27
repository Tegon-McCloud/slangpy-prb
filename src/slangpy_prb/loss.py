import slangpy as spy
import numpy as np

class L2Loss:

    def __init__(
        self,
        device: spy.Device,
        reference: spy.Texture,
    ):
        super().__init__()

        self.device = device
        self.reference = reference

        self.loss_program = self.device.load_program("shaders/l2_loss.slang", ["main"])
        self.loss_kernel = self.device.create_compute_kernel(self.loss_program)

        self.adjoint_program = self.device.load_program("shaders/l2_adjoint.slang", ["main"])
        self.adjoint_kernel = self.device.create_compute_kernel(self.adjoint_program)

        self.out_buffer = self.device.create_buffer(
            size=4,
            usage=spy.BufferUsage.unordered_access,
            label="out_buffer",
        )


    def loss(
        self,
        primal: spy.Texture,
    ) -> float:
        command_encoder = self.device.create_command_encoder()
        command_encoder.clear_buffer(self.out_buffer)
        self.loss_kernel.dispatch(
            thread_count=[self.reference.width, self.reference.height, 1],
            vars={
                "primal": primal,
                "reference": self.reference,
                "out": self.out_buffer,
            },
            command_encoder=command_encoder,
        )
        self.device.submit_command_buffer(command_encoder.finish())
        
        l = float(self.out_buffer.to_numpy().view(np.float32)[0])
        return l / (3 * self.reference.width * self.reference.height)


    def adjoint(
        self,
        command_encoder: spy.CommandEncoder,
        primal: spy.Texture,
        out: spy.Texture,
    ):
        self.adjoint_kernel.dispatch(
            thread_count=[self.reference.width, self.reference.height, 1],
            vars={
                "scale":  2.0 / (3.0 * self.reference.width * self.reference.height),
                "primal": primal,
                "reference": self.reference,
                "adjoint": out,
            },
            command_encoder=command_encoder,
        )

