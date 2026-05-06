import slangpy as spy
import numpy as np

class Loss:

    def loss(
        self,
        primal: spy.Texture,
    ) -> float: ...

    def backwards(
        self,
        command_encoder: spy.CommandEncoder,
        primal: spy.Texture,
        output: spy.Texture,
    ): ...

class L2Loss(Loss):
    def __init__(
        self,
        device: spy.Device,
        reference: spy.Texture,
    ):
        super().__init__()

        self.device = device
        self.reference = reference

        self.module = self.device.load_module("shaders/l2_loss.slang")
        
        self.loss_program = device.link_program([self.module], [self.module.entry_point("loss")])
        self.loss_pipeline = self.device.create_compute_pipeline(self.loss_program)

        self.backwards_program = device.link_program([self.module], [self.module.entry_point("backwards")])
        self.backwards_pipeline = self.device.create_compute_pipeline(self.backwards_program)

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

        compute_pass = command_encoder.begin_compute_pass()
        shader_object = compute_pass.bind_pipeline(self.loss_pipeline)
        cursor = spy.ShaderCursor(shader_object)
        cursor.reference = self.reference
        entry_cursor = cursor.find_entry_point(0)
        entry_cursor.primal = primal
        entry_cursor.output = self.out_buffer

        compute_pass.dispatch([self.reference.width, self.reference.height, 1])
        compute_pass.end()
        self.device.submit_command_buffer(command_encoder.finish())
        
        l = float(self.out_buffer.to_numpy().view(np.float32)[0])
        return l / (3 * self.reference.width * self.reference.height)


    def backwards(
        self,
        command_encoder: spy.CommandEncoder,
        primal: spy.Texture,
        output: spy.Texture,
    ):
        compute_pass = command_encoder.begin_compute_pass()
        shader_object = compute_pass.bind_pipeline(self.backwards_pipeline)
        cursor = spy.ShaderCursor(shader_object)

        cursor.reference = self.reference

        entry_cursor = cursor.find_entry_point(0)
        entry_cursor.primal = primal
        entry_cursor.gradient = output
        entry_cursor.scale = 2.0 / (3.0 * self.reference.width * self.reference.height)

        compute_pass.dispatch([self.reference.width, self.reference.height, 1])
        compute_pass.end()


class L1Loss(Loss):
    def __init__(
        self,
        device: spy.Device,
        reference: spy.Texture,
    ):
        super().__init__()

        self.device = device
        self.reference = reference

        self.module = self.device.load_module("shaders/l1_loss.slang")
        
        self.loss_program = device.link_program([self.module], [self.module.entry_point("loss")])
        self.loss_pipeline = self.device.create_compute_pipeline(self.loss_program)

        self.backwards_program = device.link_program([self.module], [self.module.entry_point("backwards")])
        self.backwards_pipeline = self.device.create_compute_pipeline(self.backwards_program)

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

        compute_pass = command_encoder.begin_compute_pass()
        shader_object = compute_pass.bind_pipeline(self.loss_pipeline)
        cursor = spy.ShaderCursor(shader_object)
        cursor.reference = self.reference
        entry_cursor = cursor.find_entry_point(0)
        entry_cursor.primal = primal
        entry_cursor.output = self.out_buffer

        compute_pass.dispatch([self.reference.width, self.reference.height, 1])
        compute_pass.end()
        self.device.submit_command_buffer(command_encoder.finish())
        
        l = float(self.out_buffer.to_numpy().view(np.float32)[0])
        return l / (3 * self.reference.width * self.reference.height)


    def backwards(
        self,
        command_encoder: spy.CommandEncoder,
        primal: spy.Texture,
        output: spy.Texture,
    ):
        compute_pass = command_encoder.begin_compute_pass()
        shader_object = compute_pass.bind_pipeline(self.backwards_pipeline)
        cursor = spy.ShaderCursor(shader_object)

        cursor.reference = self.reference

        entry_cursor = cursor.find_entry_point(0)
        entry_cursor.primal = primal
        entry_cursor.gradient = output
        entry_cursor.scale = 1.0 / (3.0 * self.reference.width * self.reference.height)

        compute_pass.dispatch([self.reference.width, self.reference.height, 1])
        compute_pass.end()