#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
biased dice designer - graphical user interface
Biased Dice Designer - Graphical User Interface

This module contains the GUI implementation of the biased dice designer, providing an intuitive interface for designing, optimizing, visualizing, and exporting dice models with a biased probability distribution.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QSlider, QPushButton, 
                           QFileDialog, QStatusBar, QGroupBox, QGridLayout,
                           QTabWidget, QFormLayout, QDoubleSpinBox, QSpinBox,
                           QLineEdit, QProgressBar, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

try:
    # when running from the project root directory
    from src.probability_models import calculate_probabilities
    from src.visualizer import visualize_dice, save_to_stl
    from src.dice_designer_cli import design_biased_dice
    from src.mesh_generation import create_blocky_mesh_from_voxels
except ImportError:
    # when running from the src directory
    from probability_models import calculate_probabilities
    from visualizer import visualize_dice, save_to_stl
    from dice_designer_cli import design_biased_dice
    from mesh_generation import create_blocky_mesh_from_voxels


class OptimizationWorker(QThread):
    """optimization worker"""
    progress_signal = pyqtSignal(int)
    completed_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
    def run(self):
        try:
            # update progress bar
            def progress_callback(progress, current_score=None, voxels=None):
                # only use the first parameter to update the progress bar
                self.progress_signal.emit(int(progress * 100))
            
            # call the optimization function, pass in the target probability and parameters
            vertices, faces, voxels, final_probs = design_biased_dice(
                self.params['target_probabilities'],
                resolution=self.params['resolution'],
                max_iterations=self.params['max_iterations'],
                progress_callback=progress_callback
            )
            
            # build the result dictionary
            result = {
                'vertices': vertices,
                'faces': faces,
                'voxels': voxels,
                'probabilities': final_probs,
                'output_dir': self.params['output_dir'],
                'output_stem': self.params['output_stem']
            }
            
            # send the completed signal with the result
            self.completed_signal.emit(result)
            
        except Exception as e:
            self.error_signal.emit(str(e))


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for 3D visualization"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)


class DiceVisualizationApp(QMainWindow):
    """dice design application main window"""
    def __init__(self):
        super().__init__()
        
        # set window title and size
        self.setWindowTitle("dice design tool")
        self.setMinimumSize(900, 700)
        
        # initialize variables
        self.voxel_model = None
        self.probabilities = np.ones(6) / 6  # default uniform distribution
        self.optimization_running = False
        self.show_walls = True  # New variable to control wall visibility
        
        # create UI components
        self.setup_ui()
        
        # set status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("ready")
        
    def setup_ui(self):
        """set up the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # main layout
        main_layout = QVBoxLayout(central_widget)
        
        # create tabs
        self.tabs = QTabWidget()
        self.design_tab = QWidget()
        self.help_tab = QWidget()
        
        self.tabs.addTab(self.design_tab, "Design Dice")
        self.tabs.addTab(self.help_tab, "Help")
        
        main_layout.addWidget(self.tabs)
        
        # initialize the design tab
        self.init_design_tab()
        
        # initialize the help tab
        self.init_help_tab()

    def init_design_tab(self):
        """initialize the design tab"""
        layout = QVBoxLayout(self.design_tab)
        
        # top horizontal layout
        top_layout = QHBoxLayout()
        
        # left parameter setting area
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # parameter setting area
        param_group = QGroupBox("parameter settings")
        param_layout = QVBoxLayout(param_group)
        
        # Recipe selection for predefined probability distributions
        recipe_group = QGroupBox("predefined probability recipes")
        recipe_layout = QVBoxLayout(recipe_group)
        
        # Create recipe buttons
        uniform_btn = QPushButton("Uniform")
        high_six_btn = QPushButton("High Six")
        high_five_six_btn = QPushButton("High Five and Six")
        linear_btn = QPushButton("Linear Increase")
        
        # Connect button signals
        uniform_btn.clicked.connect(lambda: self.apply_recipe([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]))
        high_six_btn.clicked.connect(lambda: self.apply_recipe([0.01, 0.01, 0.01, 0.01, 0.01, 0.95]))
        high_five_six_btn.clicked.connect(lambda: self.apply_recipe([0, 0, 0, 0, 0.5, 0.5]))
        linear_btn.clicked.connect(lambda: self.apply_recipe([0.05, 0.1, 0.15, 0.2, 0.25, 0.25]))
        
        # Add buttons to layout
        recipe_layout.addWidget(uniform_btn)
        recipe_layout.addWidget(high_six_btn)
        recipe_layout.addWidget(high_five_six_btn)
        recipe_layout.addWidget(linear_btn)
        
        param_layout.addWidget(recipe_group)
        
        # create target probability input - add SpinBox input and slider adjustment
        prob_group = QGroupBox("target dice face probabilities")
        prob_layout = QGridLayout(prob_group)
        
        # create input widgets for each face
        self.prob_inputs = {}
        self.sliders = []
        self.prob_labels = []
        
        face_names = ["bottom (1)", "top (6)", "back (3)", "front (4)", "left (2)", "right (5)"]
        
        for i, name in enumerate(face_names):
            # label
            label = QLabel(name)
            
            # SpinBox widget
            prob_input = QDoubleSpinBox()
            prob_input.setRange(0.01, 1.0)
            prob_input.setSingleStep(0.01)
            prob_input.setValue(1/6)  # default uniform distribution
            prob_input.setDecimals(4)
            prob_input.valueChanged.connect(lambda value, idx=i: self.update_slider_from_spinbox(idx, value))
            self.prob_inputs[i+1] = prob_input
            
            # slider widget
            slider = QSlider(Qt.Horizontal)
            slider.setRange(1, 100)
            slider.setValue(int(self.probabilities[i] * 100))
            slider.valueChanged.connect(lambda value, idx=i: self.update_probability_from_slider(idx, value))
            self.sliders.append(slider)
            
            # probability value display label
            prob_label = QLabel(f"{self.probabilities[i]:.3f}")
            self.prob_labels.append(prob_label)
            
            # add to layout
            prob_layout.addWidget(label, i, 0)
            prob_layout.addWidget(prob_input, i, 1)
            prob_layout.addWidget(slider, i, 2)
            prob_layout.addWidget(prob_label, i, 3)
        
        # add normalize button
        normalize_btn = QPushButton("normalize probabilities")
        normalize_btn.clicked.connect(self.normalize_probabilities)
        prob_layout.addWidget(normalize_btn, len(face_names), 0, 1, 4)
        
        param_layout.addWidget(prob_group)
        
        # optimization parameter setting
        optim_group = QGroupBox("optimization parameters")
        optim_layout = QFormLayout(optim_group)
        
        self.resolution_input = QSpinBox()
        self.resolution_input.setRange(10, 100)
        self.resolution_input.setValue(20)
        optim_layout.addRow("voxel resolution:", self.resolution_input)
        
        self.max_iter_input = QSpinBox()
        self.max_iter_input.setRange(100, 20000)
        self.max_iter_input.setValue(5000)
        self.max_iter_input.setSingleStep(500)
        optim_layout.addRow("max iterations:", self.max_iter_input)
        
        # output setting
        self.output_dir_input = QLineEdit("./output")
        browse_btn = QPushButton("browse...")
        browse_btn.clicked.connect(self.browse_output_dir)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_input)
        output_layout.addWidget(browse_btn)
        optim_layout.addRow("output directory:", output_layout)
        
        self.output_stem_input = QLineEdit("biased_dice")
        optim_layout.addRow("output file name prefix:", self.output_stem_input)
        
        param_layout.addWidget(optim_group)
        
        # create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        param_layout.addWidget(self.progress_bar)
        
        # action buttons
        action_layout = QHBoxLayout()
        self.start_btn = QPushButton("start optimization")
        self.start_btn.clicked.connect(self.start_optimization)
        
        self.stop_btn = QPushButton("stop")
        self.stop_btn.clicked.connect(self.stop_optimization)
        self.stop_btn.setEnabled(False)
        
        action_layout.addWidget(self.start_btn)
        action_layout.addWidget(self.stop_btn)
        param_layout.addLayout(action_layout)
        
        # add all parameter settings to the left panel
        left_layout.addWidget(param_group)
        
        # right visualization panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 3D visualization area
        viz_group = QGroupBox("3D visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Toggle walls button
        self.toggle_walls_btn = QPushButton("Hide Dice Walls")
        self.toggle_walls_btn.clicked.connect(self.toggle_walls)
        viz_layout.addWidget(self.toggle_walls_btn)
        
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        
        # No toolbar now
        viz_layout.addWidget(self.canvas)
        
        right_layout.addWidget(viz_group)
        
        # add left and right panels to the top layout
        top_layout.addWidget(left_panel, 2)
        top_layout.addWidget(right_panel, 3)
        
        # add top layout to the main layout
        layout.addLayout(top_layout)
    
    def init_help_tab(self):
        """initialize the help tab"""
        layout = QVBoxLayout(self.help_tab)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>dice design tool usage instructions</h2>
        
        <h3>design dice tab</h3>
        <p>this tab is used to design and optimize a dice model with a specific probability distribution from scratch.</p>
        <ol>
            <li>set the target probability of each face in the "target dice face probabilities" section
                <ul>
                    <li>can use the numeric input box to set the value exactly</li>
                    <li>or use the slider to adjust the probability intuitively</li>
                    <li>click the "normalize probabilities" button to ensure the sum of probabilities is 1</li>
                </ul>
            </li>
            <li>set the "voxel resolution" parameter - a larger value generates a finer model, but the calculation is slower</li>
            <li>set the "max iterations" parameter - more iterations may produce better results</li>
            <li>select the output directory and file name prefix</li>
            <li>click the "start optimization" button to start the calculation</li>
            <li>after optimization, the 3D view will display the final dice model</li>
            <li>can click the "save visualization" button to save the visualization as a picture</li>
            <li>click the "export STL model" button to export the 3D model file</li>
        </ol>

        </ul>
        
        <h3>algorithm description</h3>
        <p>this tool uses voxel optimization algorithm to gradually adjust the shape of the dice until its physical properties produce a probability distribution close to the target.
        optimization considers physical factors such as center of mass offset and solid angle, simulating the behavior of a real dice.
        
        <h3>dice face standard numbering</h3>
        <p>the standard numbering of the dice faces follows the traditional dice rules: the sum of the opposite faces is 7. that is:
        <ul>
            <li>the 1st face is on the bottom, and the 6th face is on the top</li>
            <li>the 2nd face is on the left, and the 5th face is on the right</li>
            <li>the 3rd face is on the back, and the 4th face is on the front</li>
        </ul>
        </p>
        """)
        
        layout.addWidget(help_text)

    def update_slider_from_spinbox(self, idx, value):
        """update the slider value from the numeric input box"""
        # block the signal to avoid circular updates
        self.sliders[idx].blockSignals(True)
        self.sliders[idx].setValue(int(value * 100))
        self.sliders[idx].blockSignals(False)
        
        # update the probability array and display label
        self.probabilities[idx] = value
        self.prob_labels[idx].setText(f"{value:.3f}")
        
        # normalize the other probability values
        self.normalize_all_controls()
    
    def update_probability_from_slider(self, idx, value):
        """update the probability value from the slider"""
        # convert to float
        prob_value = value / 100.0
        
        # update the numeric input box (block the signal to avoid circular updates)
        self.prob_inputs[idx+1].blockSignals(True)
        self.prob_inputs[idx+1].setValue(prob_value)
        self.prob_inputs[idx+1].blockSignals(False)
        
        # update the probability array and display label
        self.probabilities[idx] = prob_value
        self.prob_labels[idx].setText(f"{prob_value:.3f}")
        
        # normalize the other probability values
        self.normalize_all_controls()
    
    def normalize_all_controls(self):
        """normalize the values of all controls"""
        # normalize the probability array
        total = np.sum(self.probabilities)
        if total > 0:
            normalized_probs = self.probabilities / total
            
            # update all controls (block the signal to avoid circular updates)
            for i, prob in enumerate(normalized_probs):
                # only update when the probability changes significantly, avoid floating point precision issues
                if abs(self.probabilities[i] - prob) > 1e-6:
                    # update the probability array
                    self.probabilities[i] = prob
                    
                    # update the slider
                    self.sliders[i].blockSignals(True)
                    self.sliders[i].setValue(int(prob * 100))
                    self.sliders[i].blockSignals(False)
                    
                    # update the numeric input box
                    self.prob_inputs[i+1].blockSignals(True)
                    self.prob_inputs[i+1].setValue(prob)
                    self.prob_inputs[i+1].blockSignals(False)
                    
                    # update the label
                    self.prob_labels[i].setText(f"{prob:.3f}")

    def normalize_probabilities(self):
        """normalize all probability inputs so their sum is 1 (button triggered)"""
        # get values from numeric input boxes
        values = [input_field.value() for input_field in self.prob_inputs.values()]
        total = sum(values)
        
        if total > 0:
            # update all input boxes
            for i, input_field in self.prob_inputs.items():
                normalized_value = values[i-1] / total
                input_field.setValue(normalized_value)
                
                # synchronize update of slider and label (since setValue triggers the valueChanged signal, no additional operation is needed)
    
    def browse_output_dir(self):
        """open the file dialog to select the output directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "select output directory", "./",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if dir_path:
            self.output_dir_input.setText(dir_path)
    
    def start_optimization(self):
        """start the optimization process"""
        # check and create the output directory
        output_dir = self.output_dir_input.text()
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                QMessageBox.critical(self, "error", f"cannot create output directory: {str(e)}")
                return
        
        # prepare parameters
        target_probs = [self.prob_inputs[i].value() for i in range(1, 7)]
        
        # normalize
        total = sum(target_probs)
        target_probs = [p / total for p in target_probs]
        
        params = {
            'target_probabilities': target_probs,
            'resolution': self.resolution_input.value(),
            'max_iterations': self.max_iter_input.value(),
            'output_dir': output_dir,
            'output_stem': self.output_stem_input.text()
        }
        
        # create and start the worker thread
        self.worker = OptimizationWorker(params)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.completed_signal.connect(self.optimization_completed)
        self.worker.error_signal.connect(self.handle_error)
        
        # update UI status
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.optimization_running = True
        
        # start optimization
        self.worker.start()
    
    def stop_optimization(self):
        """stop the optimization process"""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            
        self.optimization_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def update_progress(self, value):
        """update the progress bar"""
        self.progress_bar.setValue(value)
    
    def optimization_completed(self, result):
        """callback when optimization is completed"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.optimization_running = False
        
        # show the result
        vertices = result.get('vertices')
        faces = result.get('faces')
        self.voxel_model = result.get('voxels')
        self.probabilities = result.get('probabilities')
        output_dir = result.get('output_dir')
        output_stem = result.get('output_stem')
        
        # save the STL file
        if vertices is not None and faces is not None and len(vertices) > 0 and len(faces) > 0:
            stl_path = os.path.join(output_dir, f"{output_stem}.stl")
            save_to_stl(vertices, faces, filename=stl_path)
        
        # load the optimization result to the interface controls
        for i, prob in enumerate(self.probabilities):
            # update the numeric input box
            self.prob_inputs[i+1].blockSignals(True)
            self.prob_inputs[i+1].setValue(prob)
            self.prob_inputs[i+1].blockSignals(False)
            
            # update the slider
            self.sliders[i].blockSignals(True)
            self.sliders[i].setValue(int(prob * 100))
            self.sliders[i].blockSignals(False)
            
            # update the label
            self.prob_labels[i].setText(f"{prob:.3f}")
        
        # update the visualization
        self.update_visualization()
        
        # show the completion message
        QMessageBox.information(
            self, 
            "optimization completed", 
            f"biased dice optimization completed!\nfinal probability distribution:\n" + 
            "\n".join([f"face {i+1}: {p:.4f}" for i, p in enumerate(self.probabilities)])
        )

    def handle_error(self, error_msg):
        """handle errors during optimization"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.optimization_running = False
        
        QMessageBox.critical(self, "optimization error", f"error during optimization:\n{error_msg}")
    
    def apply_recipe(self, probabilities):
        """Apply a predefined probability recipe"""
        # Normalize to be safe
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
            
        # Update all controls
        for i, prob in enumerate(probabilities):
            # Update the numeric input box
            self.prob_inputs[i+1].blockSignals(True)
            self.prob_inputs[i+1].setValue(prob)
            self.prob_inputs[i+1].blockSignals(False)
            
            # Update the slider
            self.sliders[i].blockSignals(True)
            self.sliders[i].setValue(int(prob * 100))
            self.sliders[i].blockSignals(False)
            
            # Update the label
            self.prob_labels[i].setText(f"{prob:.3f}")
            
            # Update the probability array
            self.probabilities[i] = prob
            
        # Update status
        recipe_summary = ", ".join([f"{p:.2f}" for p in probabilities])
        self.statusBar.showMessage(f"Applied recipe: [{recipe_summary}]")

    def toggle_walls(self):
        """Toggle the visibility of dice walls in 3D visualization"""
        self.show_walls = not self.show_walls
        
        if self.show_walls:
            self.toggle_walls_btn.setText("Hide Dice Walls")
        else:
            self.toggle_walls_btn.setText("Show Dice Walls")
            
        # Update visualization if there's a model
        if self.voxel_model is not None:
            self.update_visualization()
    
    def update_visualization(self):
        """update the 3D visualization"""
        if self.voxel_model is None:
            self.statusBar.showMessage("error: please load a model or run optimization")
            return
        
        try:
            self.statusBar.showMessage("updating visualization...")
            
            # Clear the previous graphics
            self.canvas.axes.clear()
            
            # Use the visualization function to render directly on our canvas
            # Pass our canvas axes and set show=False to avoid creating a separate window
            visualize_dice(
                self.voxel_model, 
                self.probabilities, 
                filename=None, 
                target_ax=self.canvas.axes,
                show=False,
                show_walls=self.show_walls  # Pass the walls visibility setting
            )
            
            # Refresh the canvas
            self.canvas.draw()
            
            self.statusBar.showMessage("visualization updated")
        except Exception as e:
            self.statusBar.showMessage(f"error updating visualization: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """main function"""
    app = QApplication(sys.argv)
    window = DiceVisualizationApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 