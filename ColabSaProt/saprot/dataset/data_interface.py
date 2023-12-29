import os
import yaml


# # register all available datasets through *_dataset.py files
# def construct_db():
# 	dataset_dir = os.path.dirname(__file__)
#
# 	# lists all dataset files
# 	dataset_list = []
# 	for root, _, names in os.walk(dataset_dir):
# 		for name in names:
# 			if name.endswith('_dataset.py'):
# 				sub_dirs = root.replace(dataset_dir, '').split(os.sep)
# 				dataset_list.append((sub_dirs, name[:-3]))
#
# 	# load dataset_config.yaml, controlling which dataset to load
# 	dataset_config = yaml.safe_load(open(f"{dataset_dir}/dataset_config.yaml", "r"))
#
# 	# register dataset
# 	if dataset_config["verbose"]:
# 		print("*" * 30 + f" Loading dataset " + "*" * 30)
#
# 	for sub_dirs, name in dataset_list:
# 		if name in dataset_config["datasets"]:
# 			if len(sub_dirs) > 1:
# 				cmd = f"from {'.'.join(sub_dirs)} import {name}"
# 			else:
# 				cmd = f"from . import {name}"
#
# 			exec(cmd)
#
# 			if dataset_config["verbose"]:
# 				info = f"Loaded dataset: {name}"
# 				print(f"\033[32m{info}\033[0m")
# 		else:
# 			if dataset_config["verbose"]:
# 				info = f"Skipped dataset: {name}"
# 				print(f"\033[31m{info}\033[0m")
#
# 	if dataset_config["verbose"]:
# 		print("*" * 75)
#
#
# # register function as a wrapper for all dataset
# def register_dataset(cls):
# 	dataset_dict[cls.__name__] = cls
# 	return cls
#
#
# dataset_dict = {}
# construct_db()
#
#
# class DataInterface:
# 	@classmethod
# 	def get_available_datasets(cls):
# 		return dataset_dict.keys()
#
# 	@classmethod
# 	def init_dataset(cls, dataset: str, **kwargs):
# 		"""
#
# 		Args:
# 		   dataset   : Class name of dataset you want to use. Must be in dataset_dict.keys()
# 		   **kwargs  : Kwargs for datasdet initialization
#
# 		Returns: Corresponding model
#
# 		"""
# 		assert dataset in dataset_dict.keys(), f"class {dataset} doesn't exist!"
# 		return dataset_dict[dataset](**kwargs)


########################################################################
#                             Version 2                                #
########################################################################
# register function as a wrapper for all models
def register_dataset(cls):
	global now_cls
	now_cls = cls
	return cls


now_cls = None


class DataInterface:
	@classmethod
	def init_dataset(cls, dataset_py_path: str, **kwargs):
		"""

        Args:
           dataset_py_path: Path to dataset file
           **kwargs: Kwargs for model initialization

        Returns: Corresponding model
        """
		sub_dirs = dataset_py_path.split(os.sep)
		cmd = f"from {'.' + '.'.join(sub_dirs[:-1])} import {sub_dirs[-1]}"
		exec(cmd)
		
		return now_cls(**kwargs)