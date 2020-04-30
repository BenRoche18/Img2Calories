import os
import shutil
import tkinter as tk
import cv2
from PIL import ImageTk, Image
from google_images_download import google_images_download

ROOT_DIR = os.path.join(os.path.abspath(os.sep), "Datasets", "temp")

FOOD256_DIR = os.path.join(os.path.abspath(os.sep), "Datasets", "food256")
IMAGES_DIR = os.path.join(FOOD256_DIR, "JPEGImages")
CLASS_PATH = os.path.join(FOOD256_DIR, "category.txt")

TEMP_CLASS_PATH = os.path.join(ROOT_DIR, "category.txt")
TEMP_IMAGES_DIR = os.path.join(ROOT_DIR, "images")
TEMP_LABELS_DIR = os.path.join(ROOT_DIR, "labels")

FROM_CENTER = False
SHOW_CROSSHAIR = False

def annotateImages(dir, labels):
	for path in os.listdir(dir):
		# read and display next image in directory
		img = cv2.imread(os.path.join(dir, path))
		cv2.imshow(path, img)

		# allow user to select multiple regions of interest
		boxes = cv2.selectROIs(path, img, FROM_CENTER, SHOW_CROSSHAIR)
		cv2.destroyAllWindows()

		# allow user to label each box drawn
		for box in boxes:
			root = tkinter.Tk() 

			# show image within box in window with dropdown to select label
			box_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
			box_img = ImageTk.PhotoImage(image=Image.fromarray(box_img))
  
			canvas = tkinter.Canvas(root, width=300, height=300)      
			canvas.pack()         
			canvas.create_image(20,20, anchor="nw", image=box_img)     			
			
			root.mainloop()


# returns array of classes where index represents label-1
def fetchClasses():
	with open(CLASS_PATH, 'r') as file:
		file.readline()
		class_names = [line.split('\t')[1].strip() for line in file.readlines()]

	# append classes in temp file as well
	try:
		with open(TEMP_CLASS_PATH, 'r') as file:
			for line in file.readlines():
				class_names.append(line.split('\t')[1].strip())
		return class_names
	except FileNotFoundError:
		return class_names


# creates new class and returns its label (int) or returns label if class already exists
def addClass(class_name):
	class_names = fetchClasses()
	try:
		return class_names.index(class_name) + 1
	except ValueError:
		lbl = len(class_names) + 1
		with open(TEMP_CLASS_PATH, 'a') as file:
			file.write("{0}\t{1}\n".format(lbl, class_name))
		return lbl


# returns next available unique id for image/label
def getNextID():
	# looks in temp directory first
	try:
		return max([int(filename.split('.')[0]) for filename in os.listdir(TEMP_IMAGES_DIR)]) + 1
	except (FileNotFoundError, ValueError):
		return max([int(filename.split('.')[0]) for filename in os.listdir(IMAGES_DIR)]) + 1


# add bounding box and label to corresponding img_id
def addLabel(img_id, box, lbl):
	with open(os.path.join(TEMP_LABELS_DIR, str(img_id)+".txt"), 'a') as file:
		file.write("{0} {1} {2} {3} {4}\n".format(lbl, box[0], box[1], box[2], box[3]))


# copies image at given path to images directory with given img_id as filename
def addImage(img_id, path):
	dest = os.path.join(TEMP_IMAGES_DIR, str(img_id)+".jpg")
	shutil.copy(path, dest)


### CALLBACKS BELOW ###

# fetches images with given search term from google and saves in temp directory
def fetchImages(search_term, n):
	print(search_term)

	if n <= 0 or n > 1000 or search_term == '':
		print("Invalid search parameters")
		return

	search_params = {
		"keywords": search_term,
		"limit": n,
		"format": 'jpg',
		"output_directory": ROOT_DIR,
		"print_urls": True,
	}

	response = google_images_download.googleimagesdownload()

	absolute_image_paths = response.download(search_params)

	return absolute_image_paths


# opens new window to allow user to label images in given directory
def labelImages(img_dir):
	if not os.path.isdir(img_dir) or img_dir[-1] == '\\':
		print("Invalid directory")
		return

	class_names = fetchClasses()
	for path in os.listdir(img_dir):
		img = cv2.imread(os.path.join(img_dir, path))

		# generate unique id for image
		img_id = getNextID()
		addImage(img_id, os.path.join(img_dir, path))

		skip = tk.IntVar()
		while not skip.get():
			box = cv2.selectROI(path, img, FROM_CENTER, SHOW_CROSSHAIR)

			alert = tk.Toplevel()

			# ask user to label box that corresponds to class
			tk.Label(alert, text="Select label: ").grid(row=0)
			lbl = tk.StringVar()
			tk.Spinbox(alert, textvariable=lbl, from_=1, to=len(class_names)).grid(row=0, column=1)

			def findClass(search_term):
				try:
					i = class_names.index(search_term)
					lbl.set(i+1)
				except ValueError:
					print("Class not found")

			# add text entry to search for class
			tk.Label(alert, text="Search for label: ").grid(row=1)
			search_term = tk.Entry(alert)
			search_term.grid(row=1, column=1)
			tk.Button(alert, text="Search", command=lambda: findClass(search_term.get())).grid(row=1, column=2)


			# ask user whether to discard label
			discard = tk.IntVar()
			tk.Checkbutton(alert, text="discard current label", variable=discard).grid(row=2)

			# ask user whether to skip to next image
			skip.set(1)
			tk.Checkbutton(alert, text="skip to next image", variable=skip).grid(row=3)

			def callback():
				if not discard.get():
					addLabel(img_id, box, lbl.get())
				alert.destroy()

			tk.Button(alert, text="Done", command=callback).grid(row=4)

			alert.wait_window()

		cv2.destroyAllWindows()


### CALLBACKS ABOVE ###


# displays GUI
def main():
	# create GUI window
	root = tk.Tk()
	root.title('Food Dataset')

	### WIDGETS BELOW ###

	tk.Label(root, text="IMAGE RETRIEVAL").grid(columnspan=3, pady=5)

	tk.Label(root, text="Search term for image retrieval: ").grid(row=1)
	search_term = tk.Entry(root)
	search_term.grid(row=1, column=1)

	tk.Label(root, text="Numer of images to fetch: ").grid(row=2)
	search_limit = tk.Spinbox(root, from_=0, to=1000)
	search_limit.grid(row=2, column=1)

	tk.Button(root, text="Fetch", command=lambda: fetchImages(search_term.get(), int(search_limit.get()))).grid(row=2, column=2)


	tk.Label(root, text="ANNOTATE IMAGES").grid(row=3, columnspan=3, pady=5)

	tk.Label(root, text="Select directory: ").grid(row=4)
	annotate_dir = tk.StringVar(root)
	tk.OptionMenu(root, annotate_dir, *os.listdir(ROOT_DIR)).grid(row=4, column=1)
	tk.Button(root, text="Go", command=lambda: labelImages(os.path.join(ROOT_DIR, annotate_dir.get()))).grid(row=4, column=2)


	### WIDGETS ABOVE ###

	root.mainloop()


if __name__ == '__main__':
	main()