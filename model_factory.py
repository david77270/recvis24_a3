"""Python file to instantite the model and the transform that goes with it."""
import timm
from data import data_transforms
from model import Net, PreTrained


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "pretrained":
            return PreTrained()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        if self.model_name == "pretrained":
            data_config = timm.data.resolve_model_data_config(
                self.model.pt_model
            )
            pt_transforms = timm.data.create_transform(
                **data_config,
                is_training=False
            )
            return pt_transforms
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
