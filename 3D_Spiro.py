from pathlib import Path
import zipfile, textwrap

out = Path("/mnt/data/volumetric_spirograph")
out.mkdir(parents=True, exist_ok=True)

script = r'''"""
Volumetric Fibonacci Spirograph
-------------------------------
Native Windows desktop application using GLFW + ModernGL + NumPy.
No browser, VPython, Autobahn, or PyOpenSSL.

The XY path reproduces the movement sequence in the supplied Turtle code:
- 13 outer iterations
- 37 inner iterations
- forward 55, then left 13
- an additional forward 13 at inner iteration 13
- left 55 after each inner loop

The original `i == 55` condition is retained conceptually but is unreachable
inside range(37), exactly as in the supplied program.

The 3D version creates synchronized strands. Every strand has exactly the
same XY coordinates; only Z varies. Therefore, the projection along the Z
axis preserves the original spirograph path (apart from uniform display
centering/scaling).
"""

import math
import sys

import glfw
import moderngl
import numpy as np


# ----------------------------- Configuration -----------------------------

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 850
WINDOW_TITLE = "Volumetric Spirograph | ModernGL"

OUTER_ITERATIONS = 13
INNER_ITERATIONS = 37
FORWARD_LENGTH = 55.0
SECONDARY_LENGTH = 13.0
TURN_ANGLE_DEGREES = 13.0
OUTER_TURN_DEGREES = 55.0

STRAND_COUNT = 13
DEPTH_AMPLITUDE = 95.0
DEPTH_CYCLES = 13.0

DISPLAY_RADIUS = 270.0
LINE_WIDTH = 2.0
AUTO_ROTATE_SPEED = 0.12  # radians per second

BACKGROUND = (0.006, 0.008, 0.016, 1.0)


# ------------------------------- Shaders ---------------------------------

VERTEX_SHADER = """
#version 330

in vec3 in_position;

uniform mat4 u_mvp;

out float v_z;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_z = in_position.z;
}
"""

FRAGMENT_SHADER = """
#version 330

uniform vec3 u_color;
uniform float u_alpha;

in float v_z;
out vec4 frag_color;

void main() {
    // A restrained depth-dependent brightness variation.
    float depth_shade = 0.82 + 0.18 * (0.5 + 0.5 * sin(v_z * 0.018));
    frag_color = vec4(u_color * depth_shade, u_alpha);
}
"""


# -------------------------- Matrix calculations -------------------------

def perspective(fov_degrees, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_degrees) / 2.0)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def look_at(eye, target, up):
    forward = target - eye
    forward /= np.linalg.norm(forward)

    side = np.cross(forward, up)
    side /= np.linalg.norm(side)

    corrected_up = np.cross(side, forward)

    matrix = np.identity(4, dtype=np.float32)
    matrix[0, :3] = side
    matrix[1, :3] = corrected_up
    matrix[2, :3] = -forward

    matrix[0, 3] = -np.dot(side, eye)
    matrix[1, 3] = -np.dot(corrected_up, eye)
    matrix[2, 3] = np.dot(forward, eye)
    return matrix


def camera_mvp(width, height, yaw, pitch, distance):
    aspect = max(width, 1) / max(height, 1)
    projection = perspective(45.0, aspect, 0.1, 10000.0)

    cp = math.cos(pitch)
    eye = np.array([
        distance * cp * math.sin(yaw),
        distance * math.sin(pitch),
        distance * cp * math.cos(yaw),
    ], dtype=np.float32)

    view = look_at(
        eye,
        np.zeros(3, dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    return projection @ view


# -------------------------- Original 2D path -----------------------------

def generate_original_xy():
    """
    Reproduce the supplied Turtle movement sequence as points.

    Turtle starts facing +X. `left()` increases heading counterclockwise.
    We append a point after every forward movement so no segment is lost.
    """
    x = 0.0
    y = 0.0
    heading = 0.0
    points = [(x, y)]

    for _outer in range(OUTER_ITERATIONS):
        for i in range(INNER_ITERATIONS):
            # i == 55 is intentionally retained in the condition, although
            # it cannot occur in range(37), matching the original program.
            if i == 13 or i == 55:
                angle = math.radians(heading)
                x += SECONDARY_LENGTH * math.cos(angle)
                y += SECONDARY_LENGTH * math.sin(angle)
                points.append((x, y))

            angle = math.radians(heading)
            x += FORWARD_LENGTH * math.cos(angle)
            y += FORWARD_LENGTH * math.sin(angle)
            points.append((x, y))

            heading += TURN_ANGLE_DEGREES

        heading += OUTER_TURN_DEGREES

    path = np.asarray(points, dtype=np.float32)

    # Center and uniformly scale for a comfortable viewport fit.
    # Uniform scaling does not alter the path's shape or proportions.
    path -= path.mean(axis=0, keepdims=True)
    max_radius = np.linalg.norm(path, axis=1).max()
    if max_radius > 0:
        path *= DISPLAY_RADIUS / max_radius

    return path


# ------------------------- Volumetric geometry ---------------------------

def generate_strands(xy_path, strand_count, depth_amplitude, depth_cycles):
    """
    Build a bundle of phase-shifted 3D trajectories.

    For every strand:
        x_3d[i] == x_original[i]
        y_3d[i] == y_original[i]

    Z is a smooth periodic function of path progress and strand phase.
    """
    count = len(xy_path)
    if count < 2:
        raise ValueError("The spirograph path must contain at least two points.")

    t = np.linspace(0.0, 1.0, count, dtype=np.float32)
    strands = []

    for strand_index in range(strand_count):
        phase = 2.0 * np.pi * strand_index / strand_count

        primary = np.sin(2.0 * np.pi * depth_cycles * t + phase)
        secondary = 0.22 * np.sin(
            2.0 * np.pi * (depth_cycles * 2.0) * t - phase * 0.7
        )
        z = depth_amplitude * (primary + secondary) / 1.22

        xyz = np.column_stack((xy_path, z)).astype(np.float32)
        strands.append(xyz)

    return strands


# ----------------------------- GPU resources -----------------------------

def create_gpu_strands(ctx, program, xy_path, strand_count, depth, cycles):
    strand_data = generate_strands(xy_path, strand_count, depth, cycles)
    gpu_objects = []

    for index, points in enumerate(strand_data):
        vbo = ctx.buffer(points.tobytes())
        vao = ctx.vertex_array(
            program,
            [(vbo, "3f", "in_position")],
        )
        gpu_objects.append((vbo, vao, len(points), index))

    return gpu_objects


def release_gpu_strands(gpu_objects):
    for vbo, vao, _vertex_count, _index in gpu_objects:
        vao.release()
        vbo.release()


# --------------------------------- Main ----------------------------------

def main():
    if not glfw.init():
        raise RuntimeError(
            "GLFW initialization failed. Check that a graphics driver is installed."
        )

    # Request a modern OpenGL context.
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)

    window = glfw.create_window(
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        WINDOW_TITLE,
        None,
        None,
    )

    if not window:
        glfw.terminate()
        raise RuntimeError(
            "Could not create an OpenGL 3.3 window. Update your NVIDIA driver."
        )

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # VSync

    try:
        ctx = moderngl.create_context(require=330)
    except Exception:
        glfw.destroy_window(window)
        glfw.terminate()
        raise

    print("OpenGL version:", ctx.info.get("GL_VERSION", "unknown"))
    print("GPU renderer:", ctx.info.get("GL_RENDERER", "unknown"))

    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    ctx.line_width = LINE_WIDTH

    program = ctx.program(
        vertex_shader=VERTEX_SHADER,
        fragment_shader=FRAGMENT_SHADER,
    )

    xy_path = generate_original_xy()
    strand_count = STRAND_COUNT
    depth = DEPTH_AMPLITUDE
    cycles = DEPTH_CYCLES

    gpu_strands = create_gpu_strands(
        ctx, program, xy_path, strand_count, depth, cycles
    )

    # Camera state
    yaw = 0.55
    pitch = 0.32
    distance = 850.0
    auto_rotate = False

    # Mouse state
    dragging = False
    last_mouse_x = 0.0
    last_mouse_y = 0.0

    state = {
        "yaw": yaw,
        "pitch": pitch,
        "distance": distance,
        "dragging": dragging,
        "last_x": last_mouse_x,
        "last_y": last_mouse_y,
        "auto_rotate": auto_rotate,
        "depth": depth,
        "strand_count": strand_count,
        "rebuild": False,
    }

    def update_title():
        glfw.set_window_title(
            window,
            f"Volumetric Spirograph | {state['strand_count']} strands"
            f" | depth {state['depth']:.0f}"
            f" | auto-rotate {'ON' if state['auto_rotate'] else 'OFF'}",
        )

    def mouse_button_callback(_window, button, action, _mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            state["dragging"] = action == glfw.PRESS
            if state["dragging"]:
                state["last_x"], state["last_y"] = glfw.get_cursor_pos(window)

    def cursor_position_callback(_window, xpos, ypos):
        if state["dragging"]:
            dx = xpos - state["last_x"]
            dy = ypos - state["last_y"]
            state["yaw"] += dx * 0.006
            state["pitch"] -= dy * 0.006
            state["pitch"] = max(-1.48, min(1.48, state["pitch"]))
            state["last_x"] = xpos
            state["last_y"] = ypos

    def scroll_callback(_window, _xoffset, yoffset):
        state["distance"] *= math.exp(-yoffset * 0.12)
        state["distance"] = max(350.0, min(4000.0, state["distance"]))

    def key_callback(_window, key, _scancode, action, _mods):
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

        elif key == glfw.KEY_SPACE:
            state["auto_rotate"] = not state["auto_rotate"]
            update_title()

        elif key == glfw.KEY_R:
            state["yaw"] = 0.55
            state["pitch"] = 0.32
            state["distance"] = 850.0

        elif key in (glfw.KEY_UP, glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            state["depth"] = min(300.0, state["depth"] + 10.0)
            state["rebuild"] = True
            update_title()

        elif key in (glfw.KEY_DOWN, glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            state["depth"] = max(0.0, state["depth"] - 10.0)
            state["rebuild"] = True
            update_title()

        elif key in (glfw.KEY_RIGHT, glfw.KEY_RIGHT_BRACKET):
            state["strand_count"] = min(40, state["strand_count"] + 1)
            state["rebuild"] = True
            update_title()

        elif key in (glfw.KEY_LEFT, glfw.KEY_LEFT_BRACKET):
            state["strand_count"] = max(1, state["strand_count"] - 1)
            state["rebuild"] = True
            update_title()

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    print("\nControls:")
    print("  Left mouse drag : orbit camera")
    print("  Mouse wheel     : zoom")
    print("  SPACE           : toggle automatic rotation")
    print("  UP / DOWN       : increase / decrease spatial depth")
    print("  RIGHT / LEFT    : increase / decrease strand count")
    print("  R               : reset camera")
    print("  ESC             : quit\n")
    update_title()

    previous_time = glfw.get_time()

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()

            current_time = glfw.get_time()
            delta_time = min(current_time - previous_time, 0.05)
            previous_time = current_time

            if state["auto_rotate"] and not state["dragging"]:
                state["yaw"] += AUTO_ROTATE_SPEED * delta_time

            # Rebuild strand VBOs only when a geometry parameter changes.
            if state["rebuild"]:
                release_gpu_strands(gpu_strands)
                gpu_strands = create_gpu_strands(
                    ctx,
                    program,
                    xy_path,
                    state["strand_count"],
                    state["depth"],
                    cycles,
                )
                state["rebuild"] = False

            width, height = glfw.get_framebuffer_size(window)
            if width == 0 or height == 0:
                continue

            ctx.viewport = (0, 0, width, height)
            ctx.clear(*BACKGROUND)

            mvp = camera_mvp(
                width,
                height,
                state["yaw"],
                state["pitch"],
                state["distance"],
            )

            # GLSL expects column-major matrix data.
            program["u_mvp"].write(mvp.astype("f4").T.tobytes())

            total = max(state["strand_count"] - 1, 1)

            for vbo, vao, vertex_count, strand_index in gpu_strands:
                # A red-to-crimson palette, with restrained variation.
                u = strand_index / total
                base_color = (
                    0.95,
                    0.025 + 0.045 * u,
                    0.035 + 0.10 * u,
                )

                program["u_color"].value = base_color
                program["u_alpha"].value = 0.76
                vao.render(mode=moderngl.LINE_STRIP, vertices=vertex_count)

            glfw.swap_buffers(window)

    finally:
        release_gpu_strands(gpu_strands)
        program.release()
        ctx.release()
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nApplication error:", exc)
        print(
            "\nTry updating your NVIDIA driver and verifying that "
            "ModernGL and GLFW are installed in the selected Python environment."
        )
        raise
'''

readme = r'''# Volumetric Spirograph (Native Windows / OpenGL)

A native desktop 3D visualization using Python, NumPy, GLFW and ModernGL.
It does not use VPython, a browser, WebSockets, Autobahn, or PyOpenSSL.

## 1. Install dependencies

Use the same Anaconda interpreter configured in VS Code:

```powershell
& "C:\Users\alfal\anaconda3\python.exe" -m pip install numpy moderngl glfw
```

## 2. Run

```powershell
& "C:\Users\alfal\anaconda3\python.exe" ".\volumetric_spirograph.py"
```

## Controls

- Left mouse drag: orbit camera
- Mouse wheel: zoom
- Space: toggle automatic rotation
- Up / Down: increase / decrease depth amplitude
- Right / Left: increase / decrease number of spatial strands
- R: reset camera
- Esc: exit

## Geometry

The 2D path follows the Turtle movement sequence from the original code.
The original condition `i == 55` is unreachable inside `range(37)`, so only
the extra movement at `i == 13` occurs.

All strands share exactly the same X and Y coordinates. The Z coordinate is
a deterministic periodic function with a phase offset per strand. Looking
directly along the Z axis therefore recovers the same 2D path, apart from
uniform centering and display scaling.

## Notes

- Requires an OpenGL 3.3-capable graphics driver.
- The application requests an OpenGL 3.3 core context.
- ModernGL uses the active OpenGL driver; it does not require CUDA.
- If the machine has multiple GPUs, Windows graphics settings can be used
  to prefer the NVIDIA high-performance GPU for Python.
'''

(out / "volumetric_spirograph.py").write_text(script, encoding="utf-8")
(out / "README.md").write_text(readme, encoding="utf-8")

zip_path = Path("/mnt/data/volumetric_spirograph_project.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(out / "volumetric_spirograph.py", "volumetric_spirograph/volumetric_spirograph.py")
    zf.write(out / "README.md", "volumetric_spirograph/README.md")

print("Created:")
print(out / "volumetric_spirograph.py")
print(out / "README.md")
print(zip_path)
