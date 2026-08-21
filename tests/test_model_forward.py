import torch

from evo2_distill.models.student import ArchitectureV1, ScalarStudentV1, trainable_parameter_count


def test_frozen_architecture_forward() -> None:
    architecture = ArchitectureV1()
    model = ScalarStudentV1(architecture)
    output = model(torch.zeros((2, 512), dtype=torch.long))
    assert tuple(output.shape) == (2,)
    assert architecture.receptive_field_bp == 511
    assert architecture.convolutional_layers == 13
    assert trainable_parameter_count(model) == 11_873


def test_cpu_forward_backward() -> None:
    model = ScalarStudentV1(ArchitectureV1())
    prediction = model(torch.randint(0, 5, (4, 512)))
    torch.nn.functional.huber_loss(prediction, torch.rand(4)).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

