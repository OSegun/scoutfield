"""
Perception: a real crop-disease classifier behind the pilot's interface.

Module order, which is also the order to implement them:

    datasets   PlantVillage (train) and PlantDoc (held-out shift test)
    model      EfficientNet-B0 with an MC-dropout-capable head
    train      fine-tuning, checkpointed per epoch
    metrics    ECE, MCE, NLL, Brier, reliability diagrams
    calibrate  temperature fitting and the temperature sweep
    adapter    CNNClassifier — satisfies CalibratedClassifier.observe()

``adapter`` is the hinge. Until it passes tests/test_pilot_contract.py, nothing
downstream is worth building.
"""

from scoutfield.perception.adapter import CNNClassifier  # noqa: F401
