from tkinter import *
from tkinter import filedialog
# import NeuralNetwork
import Picture
import efficiencyevaluation


class App(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title("Blood Vessels Detector")
        MainFrame(self).pack()


class MainFrame(Frame):
    def __init__(self, master, from_main=False):
        Frame.__init__(self, master)
        self.select_text_area_pic = None
        self.select_text_area_mask = None
        self.select_text_area_target = None
        self.console_text_area = None
        self.path_to_file_pic = ''
        self.path_to_file_mask = ''
        self.path_to_file_target = ''
        Label(self, text='Select picture:').pack()
        self.select_entry_pic()
        self.select_button_pic()
        Label(self, text='Select mask:').pack()
        self.select_entry_mask()
        self.select_button_mask()
        Label(self, text='Select target:').pack()
        self.select_entry_target()
        self.select_button_target()
        self.start_button()
        self.console()
        self.picture = None

    def select_entry_pic(self):
        Label(self, text='Select picture below').pack()
        self.select_text_area_pic = Text(self, width=100, height=1)
        self.select_text_area_pic.insert(INSERT, self.path_to_file_pic)
        self.select_text_area_pic.pack()

    def select_entry_mask(self):
        Label(self, text='Select mask below').pack()
        self.select_text_area_mask = Text(self, width=100, height=1)
        self.select_text_area_mask.insert(INSERT, self.path_to_file_mask)
        self.select_text_area_mask.pack()

    def select_entry_target(self):
        Label(self, text='Select target below').pack()
        self.select_text_area_target = Text(self, width=100, height=1)
        self.select_text_area_target.insert(INSERT, self.path_to_file_target)
        self.select_text_area_target.pack()

    def file_dialog_pic(self):
        self.path_to_file_pic = filedialog.askopenfilename(initialdir='.', title='Select picture')
        # self.select_text_area_pic.delete(0, 'end')
        self.select_text_area_pic.insert(INSERT, self.path_to_file_pic)

    def file_dialog_mask(self):
        self.path_to_file_mask = filedialog.askopenfilename(initialdir='.', title='Select mask')
        # self.select_text_area_mask.delete(0, 'end')
        self.select_text_area_mask.insert(INSERT, self.path_to_file_mask)

    def file_dialog_target(self):
        self.path_to_file_target = filedialog.askopenfilename(initialdir='.', title='Select target')
        # self.select_text_area_target.delete(0, 'end')
        self.select_text_area_target.insert(INSERT, self.path_to_file_target)

    def select_button_pic(self):
        Button(self, text='Browse files', command=self.file_dialog_pic).pack()

    def select_button_mask(self):
        Button(self, text='Browse files', command=self.file_dialog_mask).pack()

    def select_button_target(self):
        Button(self, text='Browse files', command=self.file_dialog_target).pack()

    def start_button(self):
        Button(self, text='Start', command=self.work).pack()

    def console(self):
        self.console_text_area = Text(self, width=100, height=10)
        self.console_text_area.configure(state="disabled")
        self.console_text_area.pack()

    def write_to_console(self, text):
        self.console_text_area.configure(state="normal")
        self.console_text_area.insert(INSERT, text + '\n')
        self.console_text_area.configure(state="disabled")

    def work(self):
        self.write_to_console('Please, keep track of Python console as well')
        self.write_to_console('reading images')
        self.write_to_console('basic image processing')
        self.picture = Picture.Picture(
            self.select_text_area_pic.get('1.0', END)[:-1],
            self.select_text_area_mask.get('1.0', END)[:-1],
            self.select_text_area_target.get('1.0', END)[:-1])
        self.picture.process_image_basic()
        self.write_to_console(
            '(accuracy,           sensitivity,        specificity)'
        )
        self.write_to_console(str(
            efficiencyevaluation.evaluation(self.picture.basic_processing_image,
                                            self.picture.target)
        ))
        self.write_to_console('advanced image processing')
        self.picture.process_image_advanced()
        self.write_to_console(
            '(accuracy,           sensitivity,        specificity)'
        )
        self.write_to_console(str(
            efficiencyevaluation.evaluation(self.picture.advanced_processing_image,
                                            self.picture.target)
        ))
        self.write_to_console('finished\nimages were saved in result/ dir')










root = App()
root.mainloop()