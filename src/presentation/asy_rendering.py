import ast
import math
import re
from dataclasses import dataclass, field
from html import escape
from types import SimpleNamespace


ASY_BLOCK_RE = re.compile(r"\[asy\]([\s\S]*?)\[/asy\]", re.IGNORECASE)
MODEL_INPUT_DIAGRAM_NOTICE = (
    "[Diagram drawing code omitted from model input. Use only facts explicitly stated "
    "in the problem text or labels. Do not measure the diagram or infer lengths, "
    "angles, positions, coordinates, or scale from the drawing unless the problem "
    "explicitly says the diagram is to scale or explicitly provides coordinates.]"
)

_COLOR_MAP = {
    "black": "#d8e6ff",
    "blue": "#60a5fa",
    "red": "#f87171",
    "green": "#4ade80",
    "gray": "#94a3b8",
    "grey": "#94a3b8",
    "white": "#f8fbff",
}

_UNIT_SCALE = {
    "bp": 1.0,
    "pt": 1.25,
    "cm": 40.0,
    "inch": 96.0,
}

_LABEL_DIRECTIONS = {
    "N": (0.0, 1.0),
    "S": (0.0, -1.0),
    "E": (1.0, 0.0),
    "W": (-1.0, 0.0),
    "NE": (0.75, 0.75),
    "NW": (-0.75, 0.75),
    "SE": (0.75, -0.75),
    "SW": (-0.75, -0.75),
}


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other):
        other_vec = coerce_vec2(other)
        return Vec2(self.x + other_vec.x, self.y + other_vec.y)

    def __sub__(self, other):
        other_vec = coerce_vec2(other)
        return Vec2(self.x - other_vec.x, self.y - other_vec.y)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vec2(self.x * float(other), self.y * float(other))
        raise TypeError("Vec2 only supports scalar multiplication.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Vec2(self.x / float(other), self.y / float(other))


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other):
        other_vec = coerce_vec3(other)
        return Vec3(self.x + other_vec.x, self.y + other_vec.y, self.z + other_vec.z)

    def __sub__(self, other):
        other_vec = coerce_vec3(other)
        return Vec3(self.x - other_vec.x, self.y - other_vec.y, self.z - other_vec.z)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vec3(self.x * float(other), self.y * float(other), self.z * float(other))
        raise TypeError("Vec3 only supports scalar multiplication.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Vec3(self.x / float(other), self.y / float(other), self.z / float(other))


def project_vec3(value):
    return Vec2(value.x + 0.58 * value.z, value.y + 0.36 * value.z)


def coerce_vec2(value):
    if isinstance(value, Vec2):
        return value
    if isinstance(value, Vec3):
        return project_vec3(value)
    if isinstance(value, tuple) and len(value) == 2:
        return Vec2(float(value[0]), float(value[1]))
    if isinstance(value, tuple) and len(value) == 3:
        return project_vec3(Vec3(float(value[0]), float(value[1]), float(value[2])))
    raise TypeError(f"Expected a 2D point, received {type(value)!r}.")


def coerce_vec3(value):
    if isinstance(value, Vec3):
        return value
    if isinstance(value, tuple) and len(value) == 3:
        return Vec3(float(value[0]), float(value[1]), float(value[2]))
    raise TypeError(f"Expected a 3D point, received {type(value)!r}.")


@dataclass(frozen=True)
class Transform:
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 1.0
    f: float = 0.0

    def apply(self, point):
        vec = coerce_vec2(point)
        return Vec2(
            self.a * vec.x + self.b * vec.y + self.c,
            self.d * vec.x + self.e * vec.y + self.f,
        )

    def __mul__(self, other):
        if isinstance(other, Transform):
            return Transform(
                a=self.a * other.a + self.b * other.d,
                b=self.a * other.b + self.b * other.e,
                c=self.a * other.c + self.b * other.f + self.c,
                d=self.d * other.a + self.e * other.d,
                e=self.d * other.b + self.e * other.e,
                f=self.d * other.c + self.e * other.f + self.f,
            )
        if isinstance(other, (Vec2, Vec3, tuple)):
            return self.apply(other)
        if isinstance(other, Renderable):
            return other.transformed(self)
        raise TypeError(f"Unsupported transform multiplication for {type(other)!r}.")


@dataclass
class SvgStyle:
    stroke: str = "#d8e6ff"
    stroke_width: float = 1.8
    fill: str = "none"
    dasharray: str | None = None
    marker_start: str | None = None
    marker_end: str | None = None
    text_color: str = "#f8fbff"
    dot_radius: float = 3.4
    font_size: float = 13.0

    def clone(self):
        return SvgStyle(
            stroke=self.stroke,
            stroke_width=self.stroke_width,
            fill=self.fill,
            dasharray=self.dasharray,
            marker_start=self.marker_start,
            marker_end=self.marker_end,
            text_color=self.text_color,
            dot_radius=self.dot_radius,
            font_size=self.font_size,
        )


class Renderable:
    def transformed(self, transform):
        raise NotImplementedError

    def bounds(self):
        raise NotImplementedError

    def svg(self, projector, style):
        raise NotImplementedError


@dataclass
class PathRenderable(Renderable):
    subpaths: list[tuple[list[Vec2], bool]] = field(default_factory=list)

    def transformed(self, transform):
        return PathRenderable(
            subpaths=[
                ([transform.apply(point) for point in points], closed)
                for points, closed in self.subpaths
            ]
        )

    def bounds(self):
        xs = []
        ys = []
        for points, _closed in self.subpaths:
            for point in points:
                xs.append(point.x)
                ys.append(point.y)
        if not xs or not ys:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def midpoint(self):
        for points, _closed in self.subpaths:
            if len(points) >= 2:
                first = points[0]
                last = points[-1]
                return Vec2((first.x + last.x) / 2.0, (first.y + last.y) / 2.0)
            if points:
                return points[0]
        return Vec2(0.0, 0.0)

    def append_value(self, value):
        if isinstance(value, PathRenderable):
            if not self.subpaths:
                self.subpaths = [(list(points), closed) for points, closed in value.subpaths]
                return self
            if not value.subpaths:
                return self
            last_points, last_closed = self.subpaths[-1]
            incoming_points, incoming_closed = value.subpaths[0]
            if last_points and incoming_points:
                stitched = list(last_points) + list(incoming_points)
                self.subpaths[-1] = (stitched, last_closed or incoming_closed)
                for extra in value.subpaths[1:]:
                    self.subpaths.append((list(extra[0]), extra[1]))
                return self
        point = coerce_vec2(value)
        if not self.subpaths:
            self.subpaths.append(([point], False))
            return self
        points, closed = self.subpaths[-1]
        if not closed:
            points.append(point)
            self.subpaths[-1] = (points, False)
        else:
            self.subpaths.append(([point], False))
        return self

    def close_last(self):
        if not self.subpaths:
            return
        points, _closed = self.subpaths[-1]
        self.subpaths[-1] = (points, True)

    def svg(self, projector, style):
        commands = []
        for points, closed in self.subpaths:
            if not points:
                continue
            start = projector(points[0])
            commands.append(f"M {start.x:.3f} {start.y:.3f}")
            for point in points[1:]:
                projected = projector(point)
                commands.append(f"L {projected.x:.3f} {projected.y:.3f}")
            if closed:
                commands.append("Z")
        if not commands:
            return ""
        attrs = [
            f'd="{escape(" ".join(commands))}"',
            f'stroke="{escape(style.stroke)}"',
            f'stroke-width="{style.stroke_width:.3f}"',
            f'fill="{escape(style.fill)}"',
            'stroke-linecap="round"',
            'stroke-linejoin="round"',
        ]
        if style.dasharray:
            attrs.append(f'stroke-dasharray="{escape(style.dasharray)}"')
        if style.marker_start:
            attrs.append(f'marker-start="url(#{style.marker_start})"')
        if style.marker_end:
            attrs.append(f'marker-end="url(#{style.marker_end})"')
        return f"<path {' '.join(attrs)} />"


@dataclass
class CircleRenderable(Renderable):
    center: Vec2
    radius: float

    def transformed(self, transform):
        center = transform.apply(self.center)
        edge = transform.apply(Vec2(self.center.x + self.radius, self.center.y))
        radius = math.hypot(edge.x - center.x, edge.y - center.y)
        return CircleRenderable(center=center, radius=radius)

    def bounds(self):
        return (
            self.center.x - self.radius,
            self.center.y - self.radius,
            self.center.x + self.radius,
            self.center.y + self.radius,
        )

    def svg(self, projector, style):
        center = projector(self.center)
        return (
            f'<circle cx="{center.x:.3f}" cy="{center.y:.3f}" r="{self.radius:.3f}" '
            f'stroke="{escape(style.stroke)}" stroke-width="{style.stroke_width:.3f}" '
            f'fill="{escape(style.fill)}" />'
        )


@dataclass
class EllipseRenderable(Renderable):
    center: Vec2
    radius_x: float
    radius_y: float

    def transformed(self, transform):
        center = transform.apply(self.center)
        edge_x = transform.apply(Vec2(self.center.x + self.radius_x, self.center.y))
        edge_y = transform.apply(Vec2(self.center.x, self.center.y + self.radius_y))
        rx = math.hypot(edge_x.x - center.x, edge_x.y - center.y)
        ry = math.hypot(edge_y.x - center.x, edge_y.y - center.y)
        return EllipseRenderable(center=center, radius_x=rx, radius_y=ry)

    def bounds(self):
        return (
            self.center.x - self.radius_x,
            self.center.y - self.radius_y,
            self.center.x + self.radius_x,
            self.center.y + self.radius_y,
        )

    def svg(self, projector, style):
        center = projector(self.center)
        return (
            f'<ellipse cx="{center.x:.3f}" cy="{center.y:.3f}" rx="{self.radius_x:.3f}" ry="{self.radius_y:.3f}" '
            f'stroke="{escape(style.stroke)}" stroke-width="{style.stroke_width:.3f}" '
            f'fill="{escape(style.fill)}" />'
        )


@dataclass
class ArcRenderable(Renderable):
    center: Vec2
    radius: float
    start_degrees: float
    end_degrees: float

    def transformed(self, transform):
        center = transform.apply(self.center)
        edge = transform.apply(Vec2(self.center.x + self.radius, self.center.y))
        radius = math.hypot(edge.x - center.x, edge.y - center.y)
        return ArcRenderable(center=center, radius=radius, start_degrees=self.start_degrees, end_degrees=self.end_degrees)

    def bounds(self):
        points = self.sample_points()
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    def sample_points(self, steps=48):
        if steps < 2:
            steps = 2
        sweep = self.end_degrees - self.start_degrees
        return [
            Vec2(
                self.center.x + self.radius * math.cos(math.radians(self.start_degrees + sweep * index / (steps - 1))),
                self.center.y + self.radius * math.sin(math.radians(self.start_degrees + sweep * index / (steps - 1))),
            )
            for index in range(steps)
        ]

    def svg(self, projector, style):
        sampled = PathRenderable(subpaths=[(self.sample_points(), False)])
        return sampled.svg(projector, style)


@dataclass
class SceneElement:
    kind: str
    payload: object
    style: SvgStyle

    def bounds(self):
        if hasattr(self.payload, "bounds"):
            return self.payload.bounds()
        if self.kind == "label":
            position = self.payload["position"]
            return position.x, position.y, position.x, position.y
        return None


class AsyRenderError(Exception):
    pass


def split_asy_blocks(text):
    if text is None:
        return []

    parts = []
    cursor = 0
    for match in ASY_BLOCK_RE.finditer(str(text)):
        if match.start() > cursor:
            parts.append(("text", str(text)[cursor:match.start()]))
        parts.append(("asy", match.group(1)))
        cursor = match.end()
    if cursor < len(str(text)):
        parts.append(("text", str(text)[cursor:]))
    if not parts:
        parts.append(("text", str(text)))
    return parts


def strip_asy_blocks_for_model_input(text, replacement=MODEL_INPUT_DIAGRAM_NOTICE):
    if text is None:
        return None

    sanitized = ASY_BLOCK_RE.sub(replacement, str(text))
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized


def render_asy_block_html(code):
    try:
        renderer = AsyRenderer()
        svg = renderer.render(code)
    except Exception:
        escaped = escape(str(code).strip())
        return (
            '<details class="asy-fallback">'
            '<summary>Diagram Source</summary>'
            f'<pre class="code-block">{escaped}</pre>'
            "</details>"
        )

    return (
        '<figure class="asy-diagram">'
        '<div class="asy-diagram__toolbar">'
        '<button type="button" class="asy-diagram__expand" aria-label="Expand diagram">Expand</button>'
        "</div>"
        f'<div class="asy-diagram__frame">{svg}</div>'
        "</figure>"
    )


class AsyRenderer:
    def __init__(self):
        self.elements = []
        self.canvas_width = None
        self.canvas_height = None
        self.unit_scale = 36.0
        self.env = {
            "pi": math.pi,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "abs": abs,
            "min": min,
            "max": max,
            "Circle": self.circle,
            "ellipse": self.ellipse,
            "arc": self.arc,
            "dir": self.dir,
            "grid": self.grid,
            "unitsquare": self.unitsquare(),
            "shift": self.shift,
            "rotate": self.rotate,
            "scale": self.scale,
            "reflect": self.reflect,
            "interp": self.interp,
            "gray": self.gray,
            "grey": self.gray,
            "linewidth": self.linewidth,
            "fontsize": self.fontsize,
            "rightanglemark": self.rightanglemark,
            "graph": self.graph,
            "Arrow": "Arrow",
            "Arrows": "Arrows",
            "true": True,
            "false": False,
            "NoneValue": None,
            "up": Vec2(0.0, 1.0),
            "down": Vec2(0.0, -1.0),
            "left": Vec2(-1.0, 0.0),
            "right": Vec2(1.0, 0.0),
            "cm": _UNIT_SCALE["cm"],
            "inch": _UNIT_SCALE["inch"],
            "pt": _UNIT_SCALE["pt"],
            "bp": _UNIT_SCALE["bp"],
            "red": _COLOR_MAP["red"],
            "blue": _COLOR_MAP["blue"],
            "green": _COLOR_MAP["green"],
            "black": _COLOR_MAP["black"],
            "white": _COLOR_MAP["white"],
            "gray_color": _COLOR_MAP["gray"],
            "grey_color": _COLOR_MAP["grey"],
        }

    def render(self, code):
        for statement in self.collect_statements(str(code)):
            self.execute_statement(statement, self.env)

        if not self.elements:
            raise AsyRenderError("No renderable elements were produced.")

        bounds = [element.bounds() for element in self.elements if element.bounds() is not None]
        if not bounds:
            raise AsyRenderError("Rendered scene had no measurable bounds.")

        min_x = min(bound[0] for bound in bounds)
        min_y = min(bound[1] for bound in bounds)
        max_x = max(bound[2] for bound in bounds)
        max_y = max(bound[3] for bound in bounds)
        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)
        pad = max(width, height) * 0.12 + 6.0
        view_min_x = min_x - pad
        view_min_y = min_y - pad
        view_width = width + pad * 2.0
        view_height = height + pad * 2.0
        desired_width = self.canvas_width or min(max(view_width * self.unit_scale / 6.0, 220.0), 460.0)
        desired_height = self.canvas_height or desired_width * (view_height / max(view_width, 1.0))
        pixel_scale = min(
            desired_width / max(view_width, 1e-6),
            desired_height / max(view_height, 1e-6),
        )

        def units_from_pixels(pixel_value):
            return float(pixel_value) / max(pixel_scale, 1e-6)

        def projector(point):
            vec = coerce_vec2(point)
            return Vec2(vec.x - view_min_x, view_height - (vec.y - view_min_y))

        body = []
        for element in self.elements:
            rendered_style = element.style.clone()
            rendered_style.stroke_width = units_from_pixels(rendered_style.stroke_width)
            rendered_style.dot_radius = units_from_pixels(rendered_style.dot_radius)
            if element.kind in {"draw", "fill"}:
                body.append(element.payload.svg(projector, rendered_style))
            elif element.kind == "dot":
                point = projector(element.payload["position"])
                body.append(
                    f'<circle cx="{point.x:.3f}" cy="{point.y:.3f}" r="{rendered_style.dot_radius:.3f}" '
                    f'fill="{escape(rendered_style.fill if rendered_style.fill != "none" else rendered_style.stroke)}" />'
                )
            elif element.kind == "label":
                payload = element.payload
                direction = payload.get("direction") or Vec2(0.0, 0.0)
                point = projector(payload["position"])
                label_offset = units_from_pixels(8.0)
                label_font_size = units_from_pixels(rendered_style.font_size)
                dx = direction.x * label_offset
                dy = -direction.y * label_offset
                anchor = "middle"
                if direction.x > 0.25:
                    anchor = "start"
                elif direction.x < -0.25:
                    anchor = "end"
                baseline = "middle"
                if direction.y > 0.25:
                    baseline = "baseline"
                elif direction.y < -0.25:
                    baseline = "hanging"
                body.append(
                    f'<text x="{point.x + dx:.3f}" y="{point.y + dy:.3f}" text-anchor="{anchor}" '
                    f'fill="{escape(rendered_style.text_color)}" font-size="{label_font_size:.3f}" dominant-baseline="{baseline}" '
                    f'font-family="Georgia, Cambria, serif">{escape(payload["text"])}</text>'
                )

        return (
            f'<svg class="asy-diagram__svg" viewBox="0 0 {view_width:.3f} {view_height:.3f}" '
            f'width="{desired_width:.0f}" height="{desired_height:.0f}" role="img" '
            f'aria-label="Rendered geometry diagram" xmlns="http://www.w3.org/2000/svg">'
            '<defs>'
            '<marker id="asy-arrow-end" markerWidth="8" markerHeight="8" refX="6.4" refY="4" orient="auto" markerUnits="strokeWidth">'
            '<path d="M 0 0 L 8 4 L 0 8 z" fill="#d8e6ff" />'
            "</marker>"
            '<marker id="asy-arrow-start" markerWidth="8" markerHeight="8" refX="1.6" refY="4" orient="auto-start-reverse" markerUnits="strokeWidth">'
            '<path d="M 8 0 L 0 4 L 8 8 z" fill="#d8e6ff" />'
            "</marker>"
            "</defs>"
            + "".join(body)
            + "</svg>"
        )

    def collect_statements(self, code):
        statements = []
        buffer = []
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        in_string = False
        string_quote = ""
        index = 0
        while index < len(code):
            char = code[index]
            next_char = code[index + 1] if index + 1 < len(code) else ""

            if not in_string and char == "/" and next_char == "/":
                index += 2
                while index < len(code) and code[index] != "\n":
                    index += 1
                continue

            buffer.append(char)

            if in_string:
                if char == "\\" and next_char:
                    buffer.append(next_char)
                    index += 2
                    continue
                if char == string_quote:
                    in_string = False
            else:
                if char in {'"', "'"}:
                    in_string = True
                    string_quote = char
                elif char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth = max(paren_depth - 1, 0)
                elif char == "[":
                    bracket_depth += 1
                elif char == "]":
                    bracket_depth = max(bracket_depth - 1, 0)
                elif char == "{":
                    brace_depth += 1
                elif char == "}":
                    brace_depth = max(brace_depth - 1, 0)

                if char == ";" and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                    statement = "".join(buffer[:-1]).strip()
                    if statement:
                        statements.append(statement)
                    buffer = []
                elif char == "}" and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                    statement = "".join(buffer).strip()
                    if statement:
                        statements.append(statement)
                    buffer = []

            index += 1

        trailing = "".join(buffer).strip()
        if trailing:
            statements.append(trailing)
        return statements

    def execute_statement(self, statement, scope):
        statement = statement.strip()
        if not statement:
            return None
        if statement.startswith("import "):
            return None
        if statement == "layer()":
            return None
        if statement.startswith("return "):
            return self.eval_value(statement[len("return "):].strip(), scope)

        function_match = re.match(r"^(real|pair)\s+([A-Za-z_]\w*)\s*\(([\s\S]*?)\)\s*\{([\s\S]*)\}$", statement)
        if function_match:
            _return_type, name, params_text, body_text = function_match.groups()
            params = [segment.strip().split()[-1] for segment in self.split_top_level(params_text, ",") if segment.strip()]
            body_statements = self.collect_statements(body_text)

            def generated_function(*args):
                local_scope = dict(scope)
                for param_name, value in zip(params, args):
                    local_scope[param_name] = value
                for body_statement in body_statements:
                    result = self.execute_statement(body_statement, local_scope)
                    if body_statement.strip().startswith("return "):
                        return result
                return None

            scope[name] = generated_function
            return None

        for_match = re.match(r"^for\s*\(([\s\S]*?);([\s\S]*?);([\s\S]*?)\)\s*\{([\s\S]*)\}$", statement)
        if for_match:
            init_text, condition_text, update_text, body_text = for_match.groups()
            if init_text.strip():
                self.execute_statement(init_text.strip(), scope)
            body_statements = self.collect_statements(body_text)
            safety = 0
            while self.eval_truthy(condition_text.strip(), scope):
                for body_statement in body_statements:
                    result = self.execute_statement(body_statement, scope)
                    if body_statement.strip().startswith("return "):
                        return result
                if update_text.strip():
                    self.execute_statement(update_text.strip(), scope)
                safety += 1
                if safety > 5000:
                    break
            return None

        if self.is_declaration(statement):
            self.handle_declaration(statement, scope)
            return None

        assignment_match = re.match(r"^([A-Za-z_]\w*(?:\[\s*[\d]+\s*\])?(?:\.[A-Za-z_]\w*)?)\s*=\s*([\s\S]+)$", statement)
        if assignment_match:
            target, expr = assignment_match.groups()
            if "." in target:
                return None
            value = self.eval_value(expr, scope)
            self.assign_target(target, value, scope)
            return value

        call_match = re.match(r"^([A-Za-z_]\w*)\s*\(([\s\S]*)\)$", statement)
        if call_match:
            name, args_text = call_match.groups()
            self.handle_command(name, args_text, scope)
            return None

        return None

    def is_declaration(self, statement):
        return bool(re.match(r"^(pair|real|path|triple|Label)(\[\])?\s+", statement))

    def handle_declaration(self, statement, scope):
        match = re.match(r"^(pair|real|path|triple|Label)(\[\])?\s+([\s\S]+)$", statement)
        if not match:
            return
        declared_type, is_array, remainder = match.groups()
        if is_array:
            scope[remainder.strip()] = {}
            return
        for segment in self.split_top_level(remainder, ","):
            piece = segment.strip()
            if not piece:
                continue
            if "=" in piece:
                target, expr = piece.split("=", 1)
                value = self.eval_value(expr.strip(), scope)
                self.assign_target(target.strip(), value, scope)
                continue
            if declared_type == "real":
                scope[piece] = 0.0
            elif declared_type == "Label":
                scope[piece] = SimpleNamespace()
            else:
                scope[piece] = None

    def assign_target(self, target, value, scope):
        indexed = re.match(r"^([A-Za-z_]\w*)\[\s*([\d]+)\s*\]$", target)
        if indexed:
            name, index = indexed.groups()
            container = scope.get(name)
            if not isinstance(container, dict):
                container = {}
                scope[name] = container
            container[int(index)] = value
            return
        scope[target] = value

    def handle_command(self, name, args_text, scope):
        args = [segment.strip() for segment in self.split_top_level(args_text, ",") if segment.strip()]
        if name == "size":
            if args:
                self.canvas_width = self.eval_numeric(args[0], scope) * 2.4
            if len(args) > 1:
                self.canvas_height = max(self.eval_numeric(args[1], scope) * 2.4, 80.0)
            return
        if name == "unitsize":
            if args:
                self.unit_scale = max(self.eval_numeric(args[0], scope), 1.0)
            return
        if name in {"draw", "fill", "filldraw", "add"}:
            renderable = self.eval_value(args[0], scope)
            style = self.parse_style(args[1:], command=name, renderable=renderable, scope=scope)
            self.elements.append(SceneElement(kind="fill" if name in {"fill", "filldraw"} else "draw", payload=renderable, style=style))
            return
        if name == "dot":
            self.handle_dot(args, scope)
            return
        if name == "label":
            self.handle_label(args, scope)
            return
        if name == "xaxis":
            self.handle_axis(args, scope, axis="x")
            return
        if name == "yaxis":
            self.handle_axis(args, scope, axis="y")
            return
        if name in {"trig_axes", "rr_cartesian_axes"}:
            self.handle_cartesian_axes(args, scope)
            return

    def handle_axis(self, args, scope, axis="x"):
        start = self.eval_numeric(args[0], scope) if args else -5.0
        end = self.eval_numeric(args[1], scope) if len(args) > 1 else 5.0
        if axis == "x":
            renderable = self.eval_path_expression(f"({start},0)--({end},0)", scope)
        else:
            renderable = self.eval_path_expression(f"(0,{start})--(0,{end})", scope)
        style = SvgStyle(stroke="#bcd4f6", stroke_width=1.5, fill="none", marker_end="asy-arrow-end")
        self.elements.append(SceneElement(kind="draw", payload=renderable, style=style))

    def handle_cartesian_axes(self, args, scope):
        if len(args) < 4:
            return
        self.handle_axis([args[0], args[1]], scope, axis="x")
        self.handle_axis([args[2], args[3]], scope, axis="y")

    def handle_dot(self, args, scope):
        label_text = None
        position_index = 0
        direction = Vec2(0.0, 0.0)
        if args and self.is_string_literal(args[0]):
            label_text = self.clean_label_text(self.eval_string(args[0]))
            position_index = 1
        position = self.anchor_position(self.eval_value(args[position_index], scope))
        style_start = position_index + 1
        if len(args) > style_start and self.looks_like_direction_arg(args[style_start]):
            direction = self.direction_from_arg(args[style_start], scope)
            style_start += 1
        style = self.parse_style(args[style_start:], command="dot", scope=scope)
        self.elements.append(SceneElement(kind="dot", payload={"position": position}, style=style))
        if label_text:
            label_style = style.clone()
            label_style.text_color = label_style.stroke
            self.elements.append(
                SceneElement(
                    kind="label",
                    payload={"text": label_text, "position": position, "direction": direction},
                    style=label_style,
                )
            )

    def handle_label(self, args, scope):
        if len(args) < 2:
            return
        text = self.clean_label_text(self.eval_string(args[0]))
        if not text:
            return
        position = self.anchor_position(self.eval_value(args[1], scope))
        direction = Vec2(0.0, 0.0)
        style_start = 2
        if len(args) > 2 and self.looks_like_direction_arg(args[2]):
            direction = self.direction_from_arg(args[2], scope)
            style_start = 3
        style = self.parse_style(args[style_start:], command="label", scope=scope)
        self.elements.append(
            SceneElement(
                kind="label",
                payload={"text": text, "position": position, "direction": direction},
                style=style,
            )
        )

    def parse_style(self, raw_tokens, command="draw", renderable=None, scope=None):
        style = SvgStyle()
        if command == "add":
            style.stroke = "rgba(188, 212, 246, 0.35)"
            style.stroke_width = 1.0
        if command == "fill":
            style.fill = "rgba(148, 163, 184, 0.25)"
            style.stroke = "none"
        if command == "filldraw":
            style.fill = "rgba(148, 163, 184, 0.25)"
        if command == "label":
            style.stroke = "#f8fbff"
            style.text_color = "#f8fbff"
        if command == "dot":
            style.fill = style.stroke
        scope = scope or self.env

        for token in raw_tokens:
            for part in self.split_top_level(token, "+"):
                piece = part.strip()
                if not piece:
                    continue
                if piece in _COLOR_MAP:
                    color = _COLOR_MAP[piece]
                    style.stroke = color
                    style.text_color = color
                    if command in {"fill", "filldraw", "dot"}:
                        style.fill = color
                    continue
                if piece.startswith(("gray(", "grey(")):
                    color = self.eval_value(piece, scope)
                    style.stroke = color
                    style.text_color = color
                    if command in {"fill", "filldraw", "dot"}:
                        style.fill = color
                    continue
                if piece == "dashed":
                    style.dasharray = "7 5"
                    continue
                if piece.startswith("linewidth("):
                    style.stroke_width = max(self.eval_numeric(piece[len("linewidth("):-1], scope), 0.8)
                    continue
                if piece.startswith("fontsize("):
                    style.font_size = max(self.eval_numeric(piece[len("fontsize("):-1], scope) * 1.25, 8.0)
                    continue
                if re.fullmatch(r"-?\d+(?:\.\d+)?\s*(bp|pt)", piece):
                    magnitude = self.eval_numeric(piece, scope)
                    style.stroke_width = max(style.stroke_width, magnitude * 0.24)
                    style.dot_radius = max(style.dot_radius, magnitude * 0.7)
                    continue
                if piece.startswith("Arrow"):
                    style.marker_end = "asy-arrow-end"
                    continue
                if piece.startswith("Arrows"):
                    style.marker_start = "asy-arrow-start"
                    style.marker_end = "asy-arrow-end"
                    continue

        if renderable and isinstance(renderable, PathRenderable):
            non_empty = any(points for points, _closed in renderable.subpaths)
            if not non_empty:
                style.stroke = "none"
        if command == "dot" and style.fill == "none":
            style.fill = style.stroke
        return style

    def split_top_level(self, text, delimiter):
        parts = []
        current = []
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        in_string = False
        quote = ""
        index = 0
        while index < len(text):
            char = text[index]
            if in_string:
                current.append(char)
                if char == "\\" and index + 1 < len(text):
                    current.append(text[index + 1])
                    index += 2
                    continue
                if char == quote:
                    in_string = False
                index += 1
                continue

            if char in {'"', "'"}:
                in_string = True
                quote = char
                current.append(char)
                index += 1
                continue
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(paren_depth - 1, 0)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(bracket_depth - 1, 0)
            elif char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth = max(brace_depth - 1, 0)

            if (
                paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
                and text.startswith(delimiter, index)
            ):
                parts.append("".join(current))
                current = []
                index += len(delimiter)
                continue

            current.append(char)
            index += 1

        parts.append("".join(current))
        return parts

    def contains_top_level(self, text, delimiter):
        pieces = self.split_top_level(text, delimiter)
        return len(pieces) > 1

    def eval_truthy(self, expr, scope):
        if not expr:
            return False
        return bool(self.eval_python_expr(expr, scope))

    def eval_numeric(self, expr, scope):
        value = self.eval_python_expr(expr, scope)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Vec2):
            return math.hypot(value.x, value.y)
        raise AsyRenderError(f"Expected numeric value from expression: {expr}")

    def eval_string(self, expr):
        try:
            return ast.literal_eval(expr)
        except Exception:
            return expr.strip().strip('"').strip("'")

    def is_string_literal(self, expr):
        expr = expr.strip()
        return len(expr) >= 2 and expr[0] == expr[-1] and expr[0] in {'"', "'"}

    def eval_value(self, expr, scope):
        expr = expr.strip()
        if not expr:
            return None
        if expr == "cycle":
            return "cycle"
        if self.contains_top_level(expr, "^^") or self.contains_top_level(expr, "--"):
            return self.eval_path_expression(expr, scope)
        value = self.eval_python_expr(expr, scope)
        if isinstance(value, tuple):
            if len(value) == 2:
                return Vec2(float(value[0]), float(value[1]))
            if len(value) == 3:
                return Vec3(float(value[0]), float(value[1]), float(value[2]))
        return value

    def eval_path_expression(self, expr, scope):
        combined = PathRenderable()
        for segment in self.split_top_level(expr, "^^"):
            tokens = self.split_top_level(segment, "--")
            current = PathRenderable()
            saw_value = False
            for token in tokens:
                piece = token.strip()
                if not piece:
                    continue
                if piece == "cycle":
                    current.close_last()
                    continue
                value = self.eval_value(piece, scope)
                saw_value = True
                if isinstance(value, (CircleRenderable, EllipseRenderable, ArcRenderable)) and len(tokens) == 1:
                    return value
                current.append_value(value)
            if saw_value:
                if not combined.subpaths:
                    combined = current
                else:
                    for subpath in current.subpaths:
                        combined.subpaths.append((list(subpath[0]), subpath[1]))
        return combined

    def eval_python_expr(self, expr, scope):
        prepared = self.prepare_expression(expr)
        safe_globals = {"__builtins__": {}}
        safe_locals = dict(self.env)
        safe_locals.update(scope)
        return eval(prepared, safe_globals, safe_locals)

    def prepare_expression(self, expr):
        prepared = str(expr).strip()
        prepared = re.sub(r"operator\s*\.\.", "NoneValue", prepared)
        prepared = prepared.replace("^", "**")
        prepared = re.sub(r"(?<=\d)\s+(?=[A-Za-z_(])", "*", prepared)
        prepared = re.sub(r"(?<=\d)(?=[A-Za-z_(])", "*", prepared)
        prepared = re.sub(r"(?<=\))\s*(?=[0-9A-Za-z_(])", "*", prepared)
        prepared = re.sub(r"(?<=\d|\))\s*(cm|inch|pt|bp)\b", r"*\1", prepared)
        return prepared

    def anchor_position(self, value):
        if isinstance(value, PathRenderable):
            return value.midpoint()
        if isinstance(value, (CircleRenderable, EllipseRenderable, ArcRenderable)):
            return value.center
        return coerce_vec2(value)

    def direction_from_token(self, token):
        dx, dy = _LABEL_DIRECTIONS.get(token.strip(), (0.0, 0.0))
        return Vec2(dx, dy)

    def looks_like_direction_arg(self, token):
        stripped = token.strip()
        if stripped in _LABEL_DIRECTIONS:
            return True
        if stripped.startswith("dir("):
            return True
        if stripped in {"up", "down", "left", "right"}:
            return True
        return False

    def direction_from_arg(self, token, scope):
        stripped = token.strip()
        if stripped in _LABEL_DIRECTIONS:
            return self.direction_from_token(stripped)
        value = self.eval_value(stripped, scope)
        return coerce_vec2(value)

    def clean_label_text(self, text):
        cleaned = str(text).strip()
        cleaned = cleaned.replace("\n", " ").strip()
        cleaned = cleaned.strip("$")
        cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\overline\{([^}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", cleaned)
        replacements = {
            r"\theta": "θ",
            r"\pi": "π",
            r"\sqrt": "√",
            r"\circ": "°",
            r"\triangle": "△",
            r"\cdot": "·",
            r"\times": "×",
            r"\leq": "≤",
            r"\geq": "≥",
        }
        for source, target in replacements.items():
            cleaned = cleaned.replace(source, target)
        cleaned = cleaned.replace("{", "").replace("}", "")
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    def circle(self, center, radius):
        return CircleRenderable(center=coerce_vec2(center), radius=float(radius))

    def ellipse(self, center, rx, ry):
        return EllipseRenderable(center=coerce_vec2(center), radius_x=float(rx), radius_y=float(ry))

    def arc(self, center, radius, start, end):
        return ArcRenderable(center=coerce_vec2(center), radius=float(radius), start_degrees=float(start), end_degrees=float(end))

    def dir(self, degrees):
        radians = math.radians(float(degrees))
        return Vec2(math.cos(radians), math.sin(radians))

    def grid(self, width, height):
        width = int(float(width))
        height = int(float(height))
        subpaths = []
        for x_value in range(width + 1):
            subpaths.append(([Vec2(float(x_value), 0.0), Vec2(float(x_value), float(height))], False))
        for y_value in range(height + 1):
            subpaths.append(([Vec2(0.0, float(y_value)), Vec2(float(width), float(y_value))], False))
        return PathRenderable(subpaths=subpaths)

    def unitsquare(self):
        return PathRenderable(subpaths=[([Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(1.0, 1.0), Vec2(0.0, 1.0)], True)])

    def shift(self, *args):
        if len(args) == 1:
            delta = coerce_vec2(args[0])
        else:
            delta = Vec2(float(args[0]), float(args[1]))
        return Transform(c=delta.x, f=delta.y)

    def rotate(self, degrees, origin=None):
        angle = math.radians(float(degrees))
        cos_value = math.cos(angle)
        sin_value = math.sin(angle)
        base = Transform(a=cos_value, b=-sin_value, d=sin_value, e=cos_value)
        if origin is None:
            return base
        origin_vec = coerce_vec2(origin)
        return Transform(c=origin_vec.x, f=origin_vec.y) * base * Transform(c=-origin_vec.x, f=-origin_vec.y)

    def scale(self, factor):
        factor = float(factor)
        return Transform(a=factor, e=factor)

    def reflect(self, first, second):
        a = coerce_vec2(first)
        b = coerce_vec2(second)
        dx = b.x - a.x
        dy = b.y - a.y
        length = math.hypot(dx, dy)
        if not length:
            return Transform()
        ux = dx / length
        uy = dy / length
        reflection = Transform(
            a=2 * ux * ux - 1,
            b=2 * ux * uy,
            d=2 * ux * uy,
            e=2 * uy * uy - 1,
        )
        return Transform(c=a.x, f=a.y) * reflection * Transform(c=-a.x, f=-a.y)

    def interp(self, first, second, factor):
        first_vec = coerce_vec2(first)
        second_vec = coerce_vec2(second)
        t_value = float(factor)
        return Vec2(
            first_vec.x + (second_vec.x - first_vec.x) * t_value,
            first_vec.y + (second_vec.y - first_vec.y) * t_value,
        )

    def gray(self, amount):
        channel = max(0, min(255, round(float(amount) * 255)))
        return f"rgb({channel}, {channel}, {channel})"

    def linewidth(self, amount):
        return float(amount)

    def fontsize(self, amount):
        return float(amount)

    def rightanglemark(self, first, vertex, third, size=20):
        a = coerce_vec2(first)
        b = coerce_vec2(vertex)
        c = coerce_vec2(third)
        marker_size = max(float(size) / 20.0, 0.35)
        ba = a - b
        bc = c - b
        len_ba = math.hypot(ba.x, ba.y) or 1.0
        len_bc = math.hypot(bc.x, bc.y) or 1.0
        u = Vec2(ba.x / len_ba, ba.y / len_ba)
        v = Vec2(bc.x / len_bc, bc.y / len_bc)
        p1 = b + u * marker_size
        p2 = p1 + v * marker_size
        p3 = b + v * marker_size
        return PathRenderable(subpaths=[([p1, p2, p3], False)])

    def graph(self, function, start, end, *extras, **kwargs):
        start_value = float(start)
        end_value = float(end)
        steps = 220
        span = end_value - start_value
        if abs(span) < 1e-9:
            span = 1.0
        points = []
        for index in range(steps):
            x_value = start_value + span * index / (steps - 1)
            y_value = function(x_value)
            if isinstance(y_value, (Vec2, Vec3, tuple)):
                points.append(coerce_vec2(y_value))
            else:
                points.append(Vec2(x_value, float(y_value)))
        return PathRenderable(subpaths=[(points, False)])
