import os
import yaml
import glob


# register all available models through *_model.py files
# def construct_model():
#     model_dir = os.path.dirname(__file__)
#
#     # lists all model files
#     model_list = []
#     for root, _, names in os.walk(model_dir):
#         for name in names:
#             if name.endswith('_model.py'):
#                 sub_dirs = root.replace(model_dir, '').split(os.sep)
#                 model_list.append((sub_dirs, name[:-3]))
#
#     # load model_config.yaml, controlling which models to be loaded
#     model_config = yaml.safe_load(open(f"{model_dir}/model_config.yaml", "r"))
#
#     if model_config["verbose"]:
#         print("*" * 30 + f" Loading model " + "*" * 30)
#
#     # register models
#     for sub_dirs, name in model_list:
#         if name in model_config["models"]:
#             if len(sub_dirs) > 1:
#                 cmd = f"from {'.'.join(sub_dirs)} import {name}"
#             else:
#                 cmd = f"from . import {name}"
#
#             exec(cmd)
#
#             if model_config["verbose"]:
#                 info = f"Loaded model: {name}"
#                 print(f"\033[32m{info}\033[0m")
#         else:
#             if model_config["verbose"]:
#                 info = f"Skipped model: {name}"
#                 print(f"\033[31m{info}\033[0m")
#
#     if model_config["verbose"]:
#         print("*" * 75)
#
#
# # register function as a wrapper for all models
# def register_model(cls):
#     model_dict[cls.__name__] = cls
#     return cls
#
#
# model_dict = {}
# construct_model()
#
#
# class ModelInterface:
#     @classmethod
#     def get_available_models(cls):
#         return model_dict.keys()
#
#     @classmethod
#     def init_model(cls, model: str, **kwargs):
#         """
#
#         Args:
#            model   : Class name of model you want to use. Must be in model_dict.keys()
#            **kwargs: Kwargs for model initialization
#
#         Returns: Corresponding model
#
#         """
#         assert model in model_dict.keys(), f"class {model} doesn't exist!"
#         return model_dict[model](**kwargs)


########################################################################
#                             Version 2                                #
########################################################################
# register function as a wrapper for all models
def register_model(cls):
    global now_cls
    now_cls = cls
    return cls


now_cls = None


class ModelInterface:
    @classmethod
    def init_model(cls, model_py_path: str, **kwargs):
        """

        Args:
            model_py_path: Py file Path of model you want to use.
           **kwargs: Kwargs for model initialization

        Returns: Corresponding model
        """
        sub_dirs = model_py_path.split(os.sep)
        cmd = f"from {'.' + '.'.join(sub_dirs[:-1])} import {sub_dirs[-1]}"
        exec(cmd)

        return now_cls(**kwargs)