import os

dataset = "food101"

if dataset == "food256":

	CLASS_PATH = os.path.join(os.path.abspath(os.sep), "Datasets", "food101", "meta", "classes.txt")

	with open(CLASS_PATH, 'r') as file:
		class_names = [line.strip() for line in file.readlines()]

	print('"' + '",\n"'.join(class_names) + '"')

else:

	CLASS_PATH = os.path.join(os.path.abspath(os.sep), "Datasets", "food101", "meta", "classes.txt")

	with open(CLASS_PATH, 'r') as file:
		class_names = [line.strip() for line in file.readlines()]

	print('"' + '",\n"'.join(class_names) + '"')