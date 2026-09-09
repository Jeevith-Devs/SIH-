"""
Interactive 3D Wireframe Gimbal Visualizer for FSOC Coarse Alignment Terminal.
Renders real-time 3D rotation of the optical telescope barrel and fork mount
using perspective vector projection directly on a Tkinter Canvas.
"""

import math
import tkinter as tk
from typing import List, Tuple, Dict, Any


class Gimbal3DWidget(tk.Frame):
    """
    Real-time 3D wireframe visualizer for the 2-axis Pan-Tilt FSOC optical head.
    Displays terminal pedestal, azimuth fork, optical barrel, and laser boresight beam.
    """

    def __init__(self, parent, width: int = 320, height: int = 240, bg: str = "#0d1117", **kwargs):
        super().__init__(parent, bg=bg, **kwargs)
        self.width = width
        self.height = height
        self.bg = bg

        self.canvas = tk.Canvas(
            self,
            width=width,
            height=height,
            bg=bg,
            highlightthickness=1,
            highlightbackground="#30363d"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.pan_deg = 0.0
        self.tilt_deg = 0.0
        self.is_locked = False

        # Camera viewing angle for 3D isometric perspective
        self.view_yaw = math.radians(-35.0)    # angled slightly to show 3D depth
        self.view_pitch = math.radians(22.0)   # look slightly downward onto terminal

        # Interactive mouse drag support
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self._last_mouse_x = 0
        self._last_mouse_y = 0

        # Pre-build 3D model geometry
        self._build_geometry()
        self.render()

    def _on_mouse_down(self, event):
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

    def _on_mouse_drag(self, event):
        dx = event.x - self._last_mouse_x
        dy = event.y - self._last_mouse_y
        self.view_yaw += dx * 0.012
        self.view_pitch = max(-math.pi / 2.3, min(math.pi / 2.3, self.view_pitch + dy * 0.012))
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y
        self.render()

    def _build_geometry(self):
        """Builds vertex and edge lists for the terminal base, fork, and optical barrel."""
        # 1. Base pedestal ring (Y = -60 to -45)
        self.base_edges: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
        n_segments = 12
        r_base = 50.0
        for i in range(n_segments):
            a1 = 2.0 * math.pi * i / n_segments
            a2 = 2.0 * math.pi * (i + 1) / n_segments
            p1_bot = (r_base * math.cos(a1), -60.0, r_base * math.sin(a1))
            p2_bot = (r_base * math.cos(a2), -60.0, r_base * math.sin(a2))
            p1_top = (r_base * 0.8 * math.cos(a1), -45.0, r_base * 0.8 * math.sin(a1))
            p2_top = (r_base * 0.8 * math.cos(a2), -45.0, r_base * 0.8 * math.sin(a2))
            self.base_edges.append((p1_bot, p2_bot))
            self.base_edges.append((p1_top, p2_top))
            self.base_edges.append((p1_bot, p1_top))

        # 2. Fork bracket arms (connected to azimuth rotation)
        # 3. Optical barrel (connected to elevation rotation)
        # Barrel will be rotated dynamically in _transform_and_draw()

    def update_angles(self, pan_deg: float, tilt_deg: float, is_locked: bool = True):
        """Update gimbal angles and trigger canvas repaint."""
        self.pan_deg = float(pan_deg)
        self.tilt_deg = float(tilt_deg)
        self.is_locked = bool(is_locked)
        self.render()

    def _project_3d(self, x: float, y: float, z: float) -> Tuple[int, int]:
        """Projects a 3D coordinate (x, y, z) into 2D canvas space with perspective."""
        # 1. Apply global observer view rotation
        # Rotate around Y (yaw)
        cos_vy, sin_vy = math.cos(self.view_yaw), math.sin(self.view_yaw)
        x1 = x * cos_vy + z * sin_vy
        y1 = y
        z1 = -x * sin_vy + z * cos_vy

        # Rotate around X (pitch)
        cos_vp, sin_vp = math.cos(self.view_pitch), math.sin(self.view_pitch)
        x2 = x1
        y2 = y1 * cos_vp - z1 * sin_vp
        z2 = y1 * sin_vp + z1 * cos_vp

        # 2. Perspective projection
        d = 280.0
        depth = max(50.0, z2 + d)
        scale = 220.0 / depth

        cx = self.width / 2.0
        cy = self.height / 2.0 + 15.0

        screen_x = int(cx + x2 * scale)
        screen_y = int(cy - y2 * scale)
        return screen_x, screen_y

    def render(self):
        """Repaints the 3D gimbal model."""
        self.canvas.delete("all")

        # Background grid watermark
        self.canvas.create_text(
            10, 12,
            text="GIMBAL 3D TELEMETRY",
            anchor="w",
            fill="#8b949e",
            font=("Segoe UI", 8, "bold")
        )

        status_color = "#3fb950" if self.is_locked else "#f85149"
        status_text = "ALIGNMENT: LOCKED" if self.is_locked else "ALIGNMENT: ACQUIRING"
        self.canvas.create_text(
            self.width - 10, 12,
            text=status_text,
            anchor="e",
            fill=status_color,
            font=("Segoe UI", 8, "bold")
        )

        # 1. Draw static pedestal
        for p1, p2 in self.base_edges:
            s1 = self._project_3d(*p1)
            s2 = self._project_3d(*p2)
            self.canvas.create_line(s1[0], s1[1], s2[0], s2[1], fill="#30363d", width=1)

        # 2. Pan rotation (Azimuth around Y axis)
        pan_rad = math.radians(self.pan_deg)
        cos_p, sin_p = math.cos(pan_rad), math.sin(pan_rad)

        def rot_pan(vx, vy, vz):
            return (vx * cos_p + vz * sin_p, vy, -vx * sin_p + vz * cos_p)

        # Draw Azimuth Turntable Ring (Y = -45)
        turntable_r = 38.0
        tt_pts = []
        for i in range(12):
            ang = 2.0 * math.pi * i / 12
            raw_pt = (turntable_r * math.cos(ang), -45.0, turntable_r * math.sin(ang))
            tt_pts.append(self._project_3d(*rot_pan(*raw_pt)))
        for i in range(12):
            p1 = tt_pts[i]
            p2 = tt_pts[(i + 1) % 12]
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#58a6ff", width=1)

        # Fork arms rising from turntable (X = -28 and +28, Y goes from -45 to 0)
        fork_color = "#79c0ff"
        left_arm_bot = rot_pan(-28.0, -45.0, 0.0)
        left_arm_top = rot_pan(-28.0, 0.0, 0.0)
        right_arm_bot = rot_pan(28.0, -45.0, 0.0)
        right_arm_top = rot_pan(28.0, 0.0, 0.0)

        s_lab = self._project_3d(*left_arm_bot)
        s_lat = self._project_3d(*left_arm_top)
        s_rab = self._project_3d(*right_arm_bot)
        s_rat = self._project_3d(*right_arm_top)

        self.canvas.create_line(s_lab[0], s_lab[1], s_lat[0], s_lat[1], fill=fork_color, width=3)
        self.canvas.create_line(s_rab[0], s_rab[1], s_rat[0], s_rat[1], fill=fork_color, width=3)
        # Fork crossbar connecting arm tops to gimbal trunnion
        self.canvas.create_line(s_lat[0], s_lat[1], s_rat[0], s_rat[1], fill="#388bfd", width=2)

        # 3. Tilt rotation (Elevation around X-axis of the fork)
        tilt_rad = math.radians(self.tilt_deg)
        cos_t, sin_t = math.cos(tilt_rad), math.sin(tilt_rad)

        def rot_barrel(bx, by, bz):
            # First tilt around X axis:
            # (bx, by * cos_t - bz * sin_t, by * sin_t + bz * cos_t)
            tx = bx
            ty = by * cos_t - bz * sin_t
            tz = by * sin_t + bz * cos_t
            # Then pan around Y axis:
            return rot_pan(tx, ty, tz)

        # 4. Optical Barrel (Hexagonal Cylinder pointing along +Z)
        barrel_r = 18.0
        barrel_back_z = -25.0
        barrel_front_z = 35.0
        barrel_color = "#f0883e"

        front_ring = []
        back_ring = []
        n_hex = 8
        for i in range(n_hex):
            ang = 2.0 * math.pi * i / n_hex
            cx = barrel_r * math.cos(ang)
            cy = barrel_r * math.sin(ang)
            front_pt = rot_barrel(cx, cy, barrel_front_z)
            back_pt = rot_barrel(cx, cy, barrel_back_z)
            front_ring.append(self._project_3d(*front_pt))
            back_ring.append(self._project_3d(*back_pt))

        # Draw barrel body struts
        for i in range(n_hex):
            p_front = front_ring[i]
            p_back = back_ring[i]
            p_front_next = front_ring[(i + 1) % n_hex]
            p_back_next = back_ring[(i + 1) % n_hex]

            self.canvas.create_line(p_front[0], p_front[1], p_front_next[0], p_front_next[1], fill=barrel_color, width=2)
            self.canvas.create_line(p_back[0], p_back[1], p_back_next[0], p_back_next[1], fill="#8c4a18", width=1)
            self.canvas.create_line(p_front[0], p_front[1], p_back[0], p_back[1], fill=barrel_color, width=1)

        # 5. Laser Boresight Line of Sight (LOS Beam)
        beam_origin = rot_barrel(0.0, 0.0, barrel_front_z)
        beam_target = rot_barrel(0.0, 0.0, barrel_front_z + 85.0)
        s_bo = self._project_3d(*beam_origin)
        s_bt = self._project_3d(*beam_target)

        beam_color = "#58a6ff" if self.is_locked else "#f0883e"
        # Glowing boresight beam
        self.canvas.create_line(s_bo[0], s_bo[1], s_bt[0], s_bt[1], fill=beam_color, width=2, dash=(6, 3))
        # Beam focal reticle circle at beam tip
        self.canvas.create_oval(s_bt[0] - 4, s_bt[1] - 4, s_bt[0] + 4, s_bt[1] + 4, outline=beam_color, width=2)

        # 6. Real-time Telemetry Readout Box at bottom
        telemetry_txt = f"AZ: {self.pan_deg:+05.1f}°  |  EL: {self.tilt_deg:+05.1f}°"
        self.canvas.create_rectangle(8, self.height - 28, self.width - 8, self.height - 8, fill="#161b22", outline="#30363d")
        self.canvas.create_text(
            self.width / 2.0,
            self.height - 18,
            text=telemetry_txt,
            fill="#c9d1d9",
            font=("Consolas", 9, "bold")
        )


def run_standalone_gimbal_window():
    """Launches a standalone interactive 3D Gimbal demonstration window."""
    root = tk.Tk()
    root.title("ASTRA-Track — 3D Pan-Tilt Gimbal Visualizer")
    root.geometry("700x650")
    root.configure(bg="#0b0f19")

    header = tk.Frame(root, bg="#131b2e", padx=16, pady=12)
    header.pack(fill=tk.X)
    tk.Label(header, text="FSOC OPTICAL TERMINAL 3D GIMBAL", font=("Segoe UI", 12, "bold"), fg="#38bdf8", bg="#131b2e").pack(anchor="w")
    tk.Label(header, text="Interactive Dual-Axis Wireframe • Drag with Mouse to Orbit 3D Camera", font=("Segoe UI", 9), fg="#94a3b8", bg="#131b2e").pack(anchor="w")

    gimbal_widget = Gimbal3DWidget(root, width=640, height=440, bg="#0b0f19")
    gimbal_widget.pack(fill=tk.BOTH, expand=True, padx=16, pady=10)

    # Control Panel
    ctrl_panel = tk.Frame(root, bg="#131b2e", padx=16, pady=12)
    ctrl_panel.pack(fill=tk.X, side=tk.BOTTOM)

    # Pan Slider
    pan_row = tk.Frame(ctrl_panel, bg="#131b2e")
    pan_row.pack(fill=tk.X, pady=3)
    tk.Label(pan_row, text="Pan (Azimuth):", font=("Segoe UI", 9, "bold"), fg="#c9d1d9", bg="#131b2e", width=14, anchor="w").pack(side=tk.LEFT)
    pan_var = tk.DoubleVar(value=0.0)
    pan_scale = tk.Scale(pan_row, from_=-45.0, to=45.0, orient=tk.HORIZONTAL, resolution=0.5, variable=pan_var,
                         bg="#1e293b", fg="#fff", highlightthickness=0, troughcolor="#0b0f19")
    pan_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

    # Tilt Slider
    tilt_row = tk.Frame(ctrl_panel, bg="#131b2e")
    tilt_row.pack(fill=tk.X, pady=3)
    tk.Label(tilt_row, text="Tilt (Elevation):", font=("Segoe UI", 9, "bold"), fg="#c9d1d9", bg="#131b2e", width=14, anchor="w").pack(side=tk.LEFT)
    tilt_var = tk.DoubleVar(value=0.0)
    tilt_scale = tk.Scale(tilt_row, from_=-45.0, to=45.0, orient=tk.HORIZONTAL, resolution=0.5, variable=tilt_var,
                          bg="#1e293b", fg="#fff", highlightthickness=0, troughcolor="#0b0f19")
    tilt_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

    # Buttons Row
    btn_row = tk.Frame(ctrl_panel, bg="#131b2e")
    btn_row.pack(fill=tk.X, pady=(6, 0))

    lock_var = tk.BooleanVar(value=True)
    chk_lock = tk.Checkbutton(btn_row, text="Alignment Lock", variable=lock_var, font=("Segoe UI", 9, "bold"),
                              bg="#131b2e", fg="#3fb950", selectcolor="#0b0f19", activebackground="#131b2e")
    chk_lock.pack(side=tk.LEFT)

    auto_scan_var = tk.BooleanVar(value=True)
    chk_scan = tk.Checkbutton(btn_row, text="Auto Scan Demo", variable=auto_scan_var, font=("Segoe UI", 9, "bold"),
                              bg="#131b2e", fg="#38bdf8", selectcolor="#0b0f19", activebackground="#131b2e")
    chk_scan.pack(side=tk.LEFT, padx=15)

    def reset_center():
        pan_var.set(0.0)
        tilt_var.set(0.0)

    tk.Button(btn_row, text="Center Gimbal", command=reset_center, font=("Segoe UI", 8, "bold"),
              bg="#1e293b", fg="#fff", bd=0, padx=10, pady=4).pack(side=tk.RIGHT)

    # Animation loop
    start_time = [0.0]

    def anim_tick():
        start_time[0] += 0.033
        t = start_time[0]
        if auto_scan_var.get():
            p = 25.0 * math.sin(t * 1.2)
            ti = 15.0 * math.sin(t * 2.4)
            pan_var.set(round(p, 1))
            tilt_var.set(round(ti, 1))

        gimbal_widget.update_angles(
            pan_deg=pan_var.get(),
            tilt_deg=tilt_var.get(),
            is_locked=lock_var.get()
        )
        root.after(30, anim_tick)

    root.after(30, anim_tick)
    root.mainloop()


if __name__ == "__main__":
    run_standalone_gimbal_window()
