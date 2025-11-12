import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import numpy as np
from isegm.inference import utils
import torch
import gui_engine
from skimage.draw import disk, circle_perimeter
from PIL import Image


DEVICE = "cuda:0"
CHECKPOINT_PATH = "C:\\Users\\schoerob.UNI-AUGSBURG\\Documents\\downloaded_weights\\skipclick_checkpoints\\full_skipclick_model_last_checkpoint.pth"
H_CANVAS_MAX, W_CANVAS_MAX = 720, 1280
H_WINDOW, W_WINDOW = H_CANVAS_MAX + 80, W_CANVAS_MAX + 80


class GUIDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.numpy_img = None
        self.my_skipclick = utils.load_is_model(CHECKPOINT_PATH, torch.device('cpu'), False)
        self.my_skipclick = self.my_skipclick.to(DEVICE)

        # The following three lines just exist for consistency of the program
        self.accumulated_points = []
        self.accumulated_labels = []
        self.mask_list = [np.zeros((H_CANVAS_MAX, W_CANVAS_MAX))]

    def initUI(self):
        layout = QVBoxLayout()

        # Create canvas for image display
        self.canvas = QGraphicsView(self)
        self.scene = QGraphicsScene(self)

        # The first image is empty (fully black)
        empty_img = np.zeros((H_CANVAS_MAX, W_CANVAS_MAX, 3), dtype=np.uint8)
        q_empty_img = self.np_to_qimage(empty_img)
        pixmap = QPixmap.fromImage(q_empty_img)
        self.image_item = QGraphicsPixmapItem(pixmap)

        # Add image to scene and scene to canvas
        self.scene.addItem(self.image_item)
        self.canvas.setScene(self.scene)

        self.canvas.setMouseTracking(True)

        # Add canvas to layout and set the layout
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.setWindowTitle('GUI Demo for Interactive Segmentation (l: Load Image, s: Save Mask, u: Undo Last Click)')
        self.setGeometry(50, 50, W_WINDOW, H_WINDOW)
        self.setFocus()
        self.canvas.setFocus()
        self.targeted_h, self.targeted_w = H_CANVAS_MAX, W_CANVAS_MAX

    def np_to_qimage(self, array):
        # Convert the NumPy array to QImage
        height, width, channel = array.shape
        bytes_per_line = 3 * width
        # Change Format_RGB888 to Format.Format_RGB888 for PyQt6
        q_image = QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return q_image

    def draw_np_image(self, np_image):
        # Remove the old image item if it exists
        if hasattr(self, 'image_item'):
            self.scene.removeItem(self.image_item)

        q_image = self.np_to_qimage(np_image)
        pixmap = QPixmap.fromImage(q_image)
        self.image_item = QGraphicsPixmapItem(pixmap)

        # Add the new image to the scene
        self.scene.addItem(self.image_item)

    def mousePressEvent(self, event):
        # Get the click position in global coordinates and map it to scene
        global_pos = event.globalPosition().toPoint()
        scene_pos = self.canvas.mapFromGlobal(global_pos)
        scene_coords = self.canvas.mapToScene(scene_pos)

        x, y = int(scene_coords.x()), int(scene_coords.y())

        if (x > 0) and (y > 0) and (x < self.targeted_w) and (y < self.targeted_h):
            # This method handles mouse press events
            if event.button() == Qt.MouseButton.LeftButton:
                print(f'Clicked coordinates: ({x}, {y}) with left mouse button')
                self.callback_left(x, y)
            elif event.button() == Qt.MouseButton.RightButton:
                print(f'Clicked coordinates: ({x}, {y}) with right mouse button')
                self.callback_right(x, y)
        else:
            print(f'Clicked coordinates: ({x}, {y}) are out of bounds')

    def keyPressEvent(self, event):
        print("Key pressed: ", event.text())
        if event.text() == "l":  # Check for 'l' key press
            dialog = QFileDialog(self)
            dialog.setNameFilter("Images (*.jpg *.jpeg *.png *.bmp *.tiff)")
            if dialog.exec():
                filename = dialog.selectedFiles()[0]
                print(filename)

                try:
                    img_loaded = Image.open(filename)

                    # Check and convert to RGB if the image is not in RGB mode
                    if img_loaded.mode != 'RGB':
                        img_loaded = img_loaded.convert('RGB')

                    # We want the image to adequately fit our canvas size
                    # 1. Store the original image dimensions
                    self.orig_w, self.orig_h = img_loaded.size

                    # 2. Determine the adequate size
                    self.targeted_h, self.targeted_w = self.scale_up_to_max(self.orig_h, self.orig_w, H_CANVAS_MAX, W_CANVAS_MAX)
                    print("Image loading. Targeted size: ", self.targeted_h, ",", self.targeted_w)

                    img_loaded = img_loaded.resize((self.targeted_w, self.targeted_h), Image.ANTIALIAS)
                    self.numpy_img = np.asarray(img_loaded)

                    gui_engine.set_image_demo(self.my_skipclick, self.numpy_img)
                    self.draw_np_image(self.numpy_img)

                    # Reset auxiliary structures
                    self.accumulated_points = []
                    self.accumulated_labels = []
                    self.mask_list = [np.zeros((self.targeted_h, self.targeted_w))]


                except FileNotFoundError:
                    print("Error: File not found.")  # Specific error handling
                except OSError as e:
                    print(f"OS error: {e}")  # Error related to image loading
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")  # Catch-all for other errors

        if event.text() == "s":  # Check for 's' key press
            dialog = QFileDialog(self)
            dialog.setNameFilter("Images (*.png)")
            if dialog.exec():
                filename = dialog.selectedFiles()[0]
                print(filename)

                print("Saving image " + filename)
                mask_to_save = np.where(self.mask_list[-1] > 0.5, 255, 0)
                mask_to_save = Image.fromarray((mask_to_save.astype(np.uint8)))
                mask_to_save = mask_to_save.resize((self.orig_w, self.orig_h), Image.NEAREST)
                mask_to_save.save(filename, format="png")

        if event.text() == "u":

            if len(self.mask_list) > 2:
                self.mask_list.pop()
                self.accumulated_points.pop()
                self.accumulated_labels.pop()
                self.mask_list[-1] = gui_engine.apply_click_demo(self.my_skipclick, self.accumulated_points, self.accumulated_labels, self.mask_list[-1])

            elif len(self.mask_list) > 1:
                self.mask_list.pop()
                self.accumulated_points.pop()
                self.accumulated_labels.pop()

            img_plot = self.plot_clicks_and_mask_for_gui(
                img=self.numpy_img, mask=self.mask_list[-1],
                accumulated_points=self.accumulated_points,
                accumulated_labels=self.accumulated_labels
            )
            img_plot = Image.fromarray(img_plot).resize((self.targeted_w, self.targeted_h), Image.ANTIALIAS)
            self.draw_np_image(np.asarray(img_plot))

    def scale_up_to_max(self, h, w, h_max, w_max):
        oversize_factor = max(h / h_max, w / w_max)
        h_new = int(h / oversize_factor)
        w_new = int(w / oversize_factor)

        return h_new, w_new

    def callback_left(self, x, y):
        print("left clicked at", x, y)
        if self.numpy_img is not None:
            self.click_executed_gui(1, x, y)

    def callback_right(self, x, y):
        print("right clicked at", x, y)
        if self.numpy_img is not None:
            self.click_executed_gui(0, x, y)


    def click_executed_gui(self, is_positive, x, y):
        self.accumulated_labels.append(is_positive)
        self.accumulated_points.append([x,y])
        self.mask_list.append(
            gui_engine.apply_click_demo(self.my_skipclick, self.accumulated_points, self.accumulated_labels, self.mask_list[-1])
        )

        img_plot = self.plot_clicks_and_mask_for_gui(
            img=self.numpy_img, mask=self.mask_list[-1],
            accumulated_points=self.accumulated_points,
            accumulated_labels=self.accumulated_labels
        )
        img_plot = Image.fromarray(img_plot).resize((self.targeted_w, self.targeted_h), Image.ANTIALIAS)
        self.draw_np_image(np.asarray(img_plot))



    def plot_clicks_and_mask_for_gui(self, img, mask, accumulated_points, accumulated_labels):
        dot_size = 4
        img = img / 255.

        binarized_mask = np.where(mask > 0.5, 1, 0).astype(np.int32)
        plot_factor = 0.5
        img_plot = np.where(
            binarized_mask[..., np.newaxis] == 1,
            img * plot_factor + (1. - plot_factor) * np.array([1., 0., 0.]),
            img
        )

        for point, label in zip(accumulated_points, accumulated_labels):  # points are [x,y] labels are 0/1
            if label == 0:
                color = np.array([1., 0., 0.])
            else:
                color = np.array([0., 1., 0.])

            img_plot = self.draw_point_autosize(img_plot, point[0], point[1], color)

        img_plot = (img_plot * 255.0).astype(np.uint8)

        return img_plot

    def draw_point_autosize(self, img, x, y, color):
        FACTOR = 0.01
        height = img.shape[0]
        x, y = int(np.floor(x)), int(np.floor(y))
        radius = int(np.floor(FACTOR * height))

        drr, dcc = disk((y, x), radius)
        crr, ccc = circle_perimeter(y, x, radius+1)

        img = np.copy(img)

        # Prune the drawings
        drr_new, dcc_new = [], []
        for dr, dc in zip(drr, dcc):
            if dr < img.shape[0] and dc < img.shape[1]:
                drr_new.append(dr)
                dcc_new.append(dc)
        drr, dcc = np.array(drr_new), np.array(dcc_new)

        crr_new, ccc_new = [], []
        for cr, cc in zip(crr, ccc):
            if cr < img.shape[0] and cc < img.shape[1]:
                crr_new.append(cr)
                ccc_new.append(cc)
        crr, ccc = np.array(crr_new), np.array(ccc_new)

        img[drr, dcc] = color
        img[crr, ccc] = np.array([1., 1., 1.])

        return img




if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = GUIDemo()
    ex.show()
    app.exec()
    #sys.exit(app.exec())