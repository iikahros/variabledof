import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, TextBox
from scipy.optimize import minimize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functools import cache

# ---- Rotation Matrices ----
@cache
def rotation_matrix_x(theta_deg):
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1]
    ])

@cache
def rotation_matrix_y(theta_deg):
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ])

@cache
def rotation_matrix_z(theta_deg):
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])

@cache
def translation_matrix(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

# ---- Joint Configuration (initial) ----
joint_config = [
    ('z', [0, 0, 50], -180, 180),
    ('y', [100, 0, 0], -90, 90),
    ('y', [100, 0, 0], -90, 90),
    ('x', [0, 50, 0], -120, 120),
    ('y', [0, 0, 50], -100, 100),
    ('y', [0, 0, 50], -100, 100),
]

# ---- Forward Kinematics ----
def compute_joint_positions_general(thetas, config):
    current_transform = np.eye(4)
    positions = [current_transform[:3, 3]]

    for theta, (axis, trans, _, _) in zip(thetas, config):
        if axis == 'x':
            R = rotation_matrix_x(theta)
        elif axis == 'y':
            R = rotation_matrix_y(theta)
        elif axis == 'z':
            R = rotation_matrix_z(theta)
        else:
            raise ValueError(f"Invalid axis '{axis}'")

        T = R @ translation_matrix(*trans)
        current_transform = current_transform @ T
        positions.append(current_transform[:3, 3])

    return np.array(positions), current_transform[:3, :3]

def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0
    return np.degrees([rx, ry, rz])

def ForwardKine(thetas):
    joints, R_tcp = compute_joint_positions_general(thetas, joint_config)
    tcp_position = joints[-1]
    tcp_orientation = rotation_matrix_to_euler_angles(R_tcp)
    return tcp_position, tcp_orientation

# ---- Tkinter Pages ----
class Page1(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.go_to_page2_button = tk.Button(self, text="Go to Page 2", command=lambda: self.controller.show_frame("Page2"))
        self.go_to_page2_button.pack(pady=20)

        self.initial_thetas = [0] * len(joint_config)
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0.05, right=0.55, top=0.95, bottom=0.05)

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.sliders = []
        self.textboxes = []

        self.position_label = tk.Label(self, text="Position (X, Y, Z):")
        self.position_label.pack(pady=10)
        self.position_textbox = tk.Entry(self)
        self.position_textbox.pack(pady=10)

        self.orientation_label = tk.Label(self, text="Orientation (Rx, Ry, Rz):")
        self.orientation_label.pack(pady=10)
        self.orientation_textbox = tk.Entry(self)
        self.orientation_textbox.pack(pady=10)

        self.update_graph_and_sliders()

    def update_graph_and_sliders(self):
        for slider in self.sliders:
            slider.ax.remove()
        for textbox in self.textboxes:
            textbox.ax.remove()
        self.sliders.clear()
        self.textboxes.clear()

        initial_thetas = [0] * len(joint_config)
        joints, R_tcp = compute_joint_positions_general(initial_thetas, joint_config)
        max_val = np.max(np.abs(joints))
        axis_limit = max_val * (3 / 2)

        self.ax.clear()
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_title(f'{len(joint_config)}DoF Arm Manipulator')
        self.ax.set_xlim(-axis_limit, axis_limit)
        self.ax.set_ylim(-axis_limit, axis_limit)
        self.ax.set_zlim(-axis_limit, axis_limit)
        self.ax.view_init(elev=45, azim=45)

        self.line, = self.ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], '-o', markersize=8, label='Robot Arm')
        self.tcp_point = self.ax.scatter(joints[-1, 0], joints[-1, 1], joints[-1, 2], color='red', s=100, label='TCP')

        for i, (axis, _, min_angle, max_angle) in enumerate(joint_config):
            slider_y = 0.75 - 0.05 * i
            ax_slider = plt.axes([0.70, slider_y, 0.18, 0.03])
            slider = Slider(ax_slider, f'Joint {i+1} ({axis.upper()})', min_angle, max_angle, valinit=0)
            slider.valtext.set_visible(False)
            self.sliders.append(slider)

            ax_text = plt.axes([0.90, slider_y, 0.03, 0.03])
            text_box = TextBox(ax_text, '', initial="0", color='lightgray')
            self.textboxes.append(text_box)

            def make_submit_callback(j=i):
                def submit(text):
                    try:
                        val = float(text)
                        if joint_config[j][2] <= val <= joint_config[j][3]:
                            self.sliders[j].set_val(val)
                        else:
                            print(f"Value out of range for Joint {j+1}")
                    except ValueError:
                        print(f"Invalid input for Joint {j+1}")
                return submit
            text_box.on_submit(make_submit_callback())

        def update(val):
            thetas = [s.val for s in self.sliders]
            new_joints, R_tcp = compute_joint_positions_general(thetas, joint_config)
            self.line.set_data(new_joints[:, 0], new_joints[:, 1])
            self.line.set_3d_properties(new_joints[:, 2])
            self.tcp_point._offsets3d = (new_joints[-1, 0:1], new_joints[-1, 1:2], new_joints[-1, 2:3])
            rx, ry, rz = rotation_matrix_to_euler_angles(R_tcp)

            self.position_textbox.delete(0, tk.END)
            self.position_textbox.insert(0, f"X: {new_joints[-1, 0]:.2f}, Y: {new_joints[-1, 1]:.2f}, Z: {new_joints[-1, 2]:.2f}")
            self.orientation_textbox.delete(0, tk.END)
            self.orientation_textbox.insert(0, f"Rx: {rx:.2f}, Ry: {ry:.2f}, Rz: {rz:.2f}")

            for i, textbox in enumerate(self.textboxes):
                textbox.set_val(f"{thetas[i]:.2f}")

            self.fig.canvas.draw_idle()

        for slider in self.sliders:
            slider.on_changed(update)

        self.fig.canvas.draw_idle()


class Page2(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.joint_entries = []
        self.axis_entries = []
        self.translation_entries = []
        self.min_entries = []
        self.max_entries = []

        self.update_joint_entries()

        self.add_button = tk.Button(self, text="Add Joint", command=self.add_joint)
        self.remove_button = tk.Button(self, text="Remove Joint", command=self.remove_joint)
        self.save_button = tk.Button(self, text="Save Configuration", command=self.save_config)

        self.add_button.grid(row=100, column=0, pady=20)
        self.remove_button.grid(row=100, column=1, pady=20)
        self.save_button.grid(row=101, column=0, columnspan=2, pady=10)

    def update_joint_entries(self):
        for widgets in self.joint_entries:
            for w in widgets:
                w.grid_forget()
        self.joint_entries.clear()
        self.axis_entries.clear()
        self.translation_entries.clear()
        self.min_entries.clear()
        self.max_entries.clear()

        for i, (axis, trans, min_a, max_a) in enumerate(joint_config):
            label = tk.Label(self, text=f"Joint {i+1} Parameters:")
            label.grid(row=i*5, column=0, padx=5, pady=5)

            axis_label = tk.Label(self, text="Axis:")
            axis_entry = tk.Entry(self)
            axis_entry.insert(0, axis)

            trans_label = tk.Label(self, text="Translation (x,y,z):")
            trans_entry = tk.Entry(self)
            trans_entry.insert(0, ','.join(map(str, trans)))

            min_label = tk.Label(self, text="Min Angle:")
            min_entry = tk.Entry(self)
            min_entry.insert(0, str(min_a))

            max_label = tk.Label(self, text="Max Angle:")
            max_entry = tk.Entry(self)
            max_entry.insert(0, str(max_a))

            axis_label.grid(row=i*5+1, column=0)
            axis_entry.grid(row=i*5+1, column=1)
            trans_label.grid(row=i*5+2, column=0)
            trans_entry.grid(row=i*5+2, column=1)
            min_label.grid(row=i*5+3, column=0)
            min_entry.grid(row=i*5+3, column=1)
            max_label.grid(row=i*5+4, column=0)
            max_entry.grid(row=i*5+4, column=1)

            self.axis_entries.append(axis_entry)
            self.translation_entries.append(trans_entry)
            self.min_entries.append(min_entry)
            self.max_entries.append(max_entry)
            self.joint_entries.append((label, axis_label, axis_entry, trans_label, trans_entry, min_label, min_entry, max_label, max_entry))

    def add_joint(self):
        joint_config.append(('z', [0, 0, 50], -180, 180))
        self.update_joint_entries()

    def remove_joint(self):
        if len(joint_config) > 1:
            joint_config.pop()
            self.update_joint_entries()

    def save_config(self):
        global joint_config
        new_config = []
        for i in range(len(self.axis_entries)):
            axis = self.axis_entries[i].get().lower()
            translation = list(map(float, self.translation_entries[i].get().split(',')))
            min_angle = float(self.min_entries[i].get())
            max_angle = float(self.max_entries[i].get())
            new_config.append((axis, translation, min_angle, max_angle))

        joint_config = new_config
        print("Updated joint_config:", joint_config)
        self.controller.frames["Page1"].update_graph_and_sliders()
        self.controller.show_frame("Page1")


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Robot Arm Simulation")
        self.geometry("1200x900")
        self.frames = {}

        for F in (Page1, Page2):
            page_name = F.__name__
            frame = F(parent=self, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame('Page1')

    def show_frame(self, page_name):
        self.frames[page_name].tkraise()

# Run the application
if __name__ == "__main__":
    app = App()
    app.mainloop()
