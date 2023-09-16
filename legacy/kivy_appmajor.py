import kivy
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
Window.clearcolor = (0.1, 0.1, 0.1, 1)
import librosa
from kivy_deeplearning_file import predicto

class AudioMFCCApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.icon='tub_app_logo.jpg'
        self.file_chooser = FileChooserIconView()
        self.file_chooser.path = '/'
        self.file_chooser.filters = ['*.wav', '*.mp3']
        self.file_chooser.bind(path=self.update_path_label)
        self.path_label = Label(text=self.file_chooser.path,size_hint=(1, .3),font_size=20)
        self.mfcc_button = Button(text='Choose Audio', on_press=self.get_mfcc, size_hint=(.5, .1),background_color =(1, 0.5, 0.1, 1),pos_hint = {'center_x': .5})
        self.mfcc_label = Label(text='Output will be displayed here',font_size=20)
        layout.add_widget(self.path_label)
        layout.add_widget(self.file_chooser)
        layout.add_widget(self.mfcc_button)
        layout.add_widget(self.mfcc_label)
        return layout

    def update_path_label(self, *args):
        self.path_label.text = self.file_chooser.path

    def get_mfcc(self, obj):
        selected_file = self.file_chooser.selection

        if selected_file:
            try:
                transcript=predicto(selected_file[0])
                self.mfcc_label.text = transcript
            except Exception as e:
                print(e)
                self.mfcc_label.text = str(e)
        else:
            self.mfcc_label.text = 'Please choose a file'
if __name__ == '__main__':
    AudioMFCCApp().run()
