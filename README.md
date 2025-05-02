
Robot Arm Simulation GUI
========================

This is a Python-based GUI application that simulates a configurable multi-DOF robotic arm using Tkinter and Matplotlib. 
It allows users to visualize forward kinematics in 3D, adjust joint angles via sliders or text input, and dynamically 
configure the robot’s joint setup.

Features
--------
- 3D visualization of robot arm and TCP (Tool Center Point)
- Real-time slider controls for each joint
- Text-based inputs for direct angle entry
- Euler orientation output of TCP
- Dynamic joint configuration editor (add, remove, modify joints)
- Tkinter multi-page GUI with page switching
- Rotation and translation matrix-based kinematics
- Forward kinematics support with live output of position and orientation


Technologies Used
-----------------
- tkinter – GUI framework
- matplotlib – 3D plotting and sliders
- numpy – Matrix computations
- scipy – Optimization (for future inverse kinematics features)
- functools.cache – Efficient matrix reuse

Joint Configuration Format
--------------------------

Each joint is defined by:

- Axis of rotation: 'x', 'y', or 'z'
- Translation vector: [x, y, z] offset after rotation
- Min/Max joint angles: in degrees

You can edit these in the Page 2 of the application.

Example:

    ('y', [100, 0, 0], -90, 90)

Forward Kinematics
------------------

The simulation computes the position and orientation of each joint by applying:

1. Rotation (based on joint axis and angle)
2. Translation (offset)
3. Chaining all transformations to compute the final TCP pose

Orientation is output as Euler angles (Rx, Ry, Rz) in degrees.

Future Improvements
-------------------

- Inverse Kinematics (IK) solver integration
- Save/load joint configurations from file
- Workspace limits
- Improved UI responsiveness and usability
