"""Safe client for the legacy Kinova-side ToolRGS TCP receiver."""

from dataclasses import dataclass
import math
import re
import socket
from typing import Dict, Iterable, Mapping, Optional, Sequence

from toolrgs.registry import ROBOT_CLIENTS


TIER_DEPTH: Dict[str, int] = {"L1": -1, "L2": 0, "L3": 1}
TOOL_TIERS: Dict[str, str] = {
    "clip": "L1", "marker": "L1", "sponge": "L2", "tape": "L2",
    "l-hex key": "L1", "t-hex key": "L1", "hex key": "L1",
    "box": "L3", "screwdriver": "L1", "spool": "L2",
    "tape measure": "L1", "mallet": "L3", "clamps": "L2",
    "wrench": "L1", "crimp tool": "L2", "pliers": "L2",
    "file": "L1", "ruler": "L2", "scissors": "L1", "cable": "L1",
    "nut": "L1", "screw": "L1", "stapler": "L1",
}
TOOL_PATTERNS = (
    ("tape measure", (r"\btape\s*-?\s*measure\b", r"\bmeasuring\s+tape\b")),
    ("crimp tool", (r"\bcrimp(?:ing)?\s*-?\s*(?:tool|pliers)\b", r"\bcrimper\b")),
    ("l-hex key", (r"\bl[-\s]*(?:shaped?\s*)?hex\s*-?\s*key\b",)),
    ("t-hex key", (r"\bt[-\s]*(?:handle\s*)?hex\s*-?\s*key\b",)),
    ("hex key", (r"\bhex\s*-?\s*key\b", r"\ballen\s*(?:key|wrench)\b")),
    ("screwdriver", (r"\bscrewdriver\b", r"\bphillips?\b", r"\bflat\s*head\b")),
    ("clamps", (r"\bclamps?\b", r"\b[gc]\s*-?\s*clamp\b")),
    ("tape", (r"\btape(?!\s*-?\s*measure)\b",)),
    ("clip", (r"\bclip\b", r"\bbinder\s+clip\b")),
    ("mallet", (r"\bmallet\b", r"\bdead\s+blow\s+hammer\b")),
    ("marker", (r"\bmarker\b", r"\bsharpie\b", r"\bhighlighter\b")),
    ("sponge", (r"\bsponge\b", r"\bscrubber\b")),
    ("spool", (r"\bspool\b", r"\breel\b", r"\bbobbin\b")),
    ("wrench", (r"\bwrench\b", r"\bspanner\b")),
    ("box", (r"\bbox(?:es)?\b", r"\bcontainer\b", r"\bcarton\b")),
    ("pliers", (r"\bpliers?\b", r"\btongs\b")),
    ("file", (r"\bfiles?\b", r"\brasps?\b")),
    ("ruler", (r"\bruler\b", r"\bstraightedge\b")),
    ("scissors", (r"\bscissors\b", r"\bshears\b")),
    ("cable", (r"\bcable\b", r"\bcord\b", r"\bwire\b")),
    ("nut", (r"\bnuts?\b",)),
    ("screw", (r"\bscrews?\b", r"\bbolts?\b")),
    ("stapler", (r"\bstapler\b", r"\bstaple\s+gun\b")),
)


def find_tool_classes(text: str):
    """Return ordered, non-duplicated 22-class matches from a prompt."""
    matches = []
    for class_name, patterns in TOOL_PATTERNS:
        if any(re.search(pattern, str(text), flags=re.IGNORECASE) for pattern in patterns):
            matches.append(class_name)
    return matches


def semantic_depth(
    text: str,
    default: int = 0,
    class_tiers: Optional[Mapping[str, str]] = None,
    policy: str = "max",
) -> int:
    """Resolve the 22-class semantic tier used by the robot receiver."""
    tiers = dict(TOOL_TIERS)
    if class_tiers:
        tiers.update({str(key).casefold(): str(value).upper() for key, value in class_tiers.items()})
    matches = []
    for class_name in find_tool_classes(text):
        tier = tiers.get(class_name)
        if tier not in TIER_DEPTH:
            raise ValueError(f"Invalid semantic depth tier for {class_name}: {tier}")
        matches.append(TIER_DEPTH[tier])
    if not matches:
        return int(default)
    if policy == "max":
        return max(matches)
    if policy == "min":
        return min(matches)
    if policy == "first":
        return matches[0]
    raise ValueError("robot.depth_policy.multiple_matches must be max, min, or first")


def _wire_number(value: float) -> str:
    rounded = round(float(value), 3)
    return str(int(rounded)) if rounded.is_integer() else f"{rounded:.3f}".rstrip("0")


@dataclass(frozen=True)
class GraspCommand:
    """Pixel-space grasp command understood by the existing lab receiver."""

    x: float
    y: float
    theta: float
    width: float
    depth: int

    def validate(self) -> None:
        values: Iterable[float] = (self.x, self.y, self.theta, self.width, self.depth)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"Grasp command contains a non-finite value: {self}")
        if self.width <= 0:
            raise ValueError(f"Grasp width must be positive: {self.width}")

    def to_wire(self) -> bytes:
        self.validate()
        fields = (self.x, self.y, self.theta, self.width, self.depth)
        return ("{" + ", ".join(_wire_number(value) for value in fields) + "}\n").encode(
            "ascii"
        )

    def validate_limits(self, limits: Mapping[str, Sequence[float]]) -> None:
        self.validate()
        for field in ("x", "y", "theta", "width", "depth"):
            bounds = limits.get(field)
            if bounds is None or len(bounds) != 2:
                raise ValueError(f"Robot limit {field!r} must contain [minimum, maximum]")
            minimum, maximum = float(bounds[0]), float(bounds[1])
            value = float(getattr(self, field))
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"Grasp command {field}={value:g} is outside [{minimum:g}, {maximum:g}]"
                )


class LegacyTCPGraspClient:
    """Explicit-connect TCP sender; it never sends while merely connecting."""

    def __init__(self, host: str, port: int = 3000, timeout_s: float = 2.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._socket: Optional[socket.socket] = None

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        if self.connected:
            return
        connection = socket.create_connection(
            (self.host, self.port), timeout=self.timeout_s
        )
        connection.settimeout(self.timeout_s)
        self._socket = connection

    def send(self, command: GraspCommand) -> None:
        if not self.connected:
            raise RuntimeError("Robot receiver is not connected")
        try:
            self._socket.sendall(command.to_wire())
        except OSError:
            self.close()
            raise

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


ROBOT_CLIENTS.register_module(
    LegacyTCPGraspClient,
    name="legacy_tcp",
    aliases=("kinova_tcp", "tcp"),
)
ROBOT_CLIENT_REGISTRY = ROBOT_CLIENTS.module_dict


def build_robot_client(cfg: Mapping[str, object]):
    """Build a robot transport without connecting or sending anything."""
    component_type = cfg.get("type", "legacy_tcp")
    try:
        client_class = ROBOT_CLIENTS.require(component_type)
    except KeyError as exc:
        available = ", ".join(sorted(ROBOT_CLIENTS.keys()))
        raise ValueError(
            f"Unknown robot client {component_type!r}; available: {available}"
        ) from exc
    return client_class(
        host=cfg["host"],
        port=cfg.get("port", 3000),
        timeout_s=cfg.get("timeout_s", 2.0),
    )
