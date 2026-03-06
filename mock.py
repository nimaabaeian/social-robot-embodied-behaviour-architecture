"""
mock.py – Mock publisher for testing downstream modules.

Mimics two output ports that other modules subscribe to:
  1. /alwayson/stm/context:o   (yarp.Port / Bottle)
       Bottle: Int32 episode_id | Int32 chunk | Int8 label
       → mirrors ShortTermMemoryV2.publish_episode_context()

  2. /interactionManager/hunger:o  (BufferedPortBottle)
       Bottle: String hs  (one of "HS1" | "HS2" | "HS3")
       → mirrors InteractionManagerModule.updateModule() hunger publish

Usage:
    python mock.py
    python mock.py --context-period 2.0 --hs-period 1.0
    python mock.py --episode-id 42 --chunk 3 --label 1 --hs HS2
"""

import argparse
import sys
import time

import yarp


class MockPublisher(yarp.RFModule):
    """
    Single RFModule that publishes mock data on:
      - /alwayson/stm/context:o
      - /interactionManager/hunger:o
    """

    # Port names match the real modules exactly
    CONTEXT_PORT_NAME = "/alwayson/stm/context:o"
    HUNGER_PORT_NAME  = "/interactionManager/hunger:o"

    # Cycling HS states (same sequence the real HungerModel produces)
    HS_STATES = ["HS1", "HS2", "HS3"]

    def __init__(self) -> None:
        yarp.RFModule.__init__(self)

        # --- configurable via RF params / CLI ---
        self.context_period: float = 2.0   # seconds between context publishes
        self.hs_period: float      = 1.0   # seconds between HS publishes
        self.episode_id: int       = 0
        self.chunk: int            = -1
        self.label: int            = 0
        self.hs_index: int         = 0     # cycles through HS_STATES
        self.fixed_hs: str         = ""    # if set, always publish this HS

        # --- YARP ports ---
        self.context_port: yarp.Port               = yarp.Port()
        self.hunger_port:  yarp.BufferedPortBottle = yarp.BufferedPortBottle()

        # --- internal timing ---
        self._last_context_ts: float = 0.0
        self._last_hs_ts: float      = 0.0

    # ------------------------------------------------------------------
    # RFModule interface
    # ------------------------------------------------------------------

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        """Open ports and read optional ResourceFinder parameters."""
        # Optional RF overrides
        if rf.check("context-period"):
            self.context_period = rf.find("context-period").asFloat64()
        if rf.check("hs-period"):
            self.hs_period = rf.find("hs-period").asFloat64()
        if rf.check("episode-id"):
            self.episode_id = rf.find("episode-id").asInt32()
        if rf.check("chunk"):
            self.chunk = rf.find("chunk").asInt32()
        if rf.check("label"):
            self.label = rf.find("label").asInt32()
        if rf.check("hs"):
            self.fixed_hs = rf.find("hs").asString()
            if self.fixed_hs not in self.HS_STATES:
                print(f"[MockPublisher] WARNING: unknown HS value '{self.fixed_hs}', cycling instead")
                self.fixed_hs = ""

        # Open context port
        if not self.context_port.open(self.CONTEXT_PORT_NAME):
            print(f"[MockPublisher] ERROR: failed to open {self.CONTEXT_PORT_NAME}")
            return False
        print(f"[MockPublisher] Port open: {self.CONTEXT_PORT_NAME}")

        # Open hunger port
        if not self.hunger_port.open(self.HUNGER_PORT_NAME):
            print(f"[MockPublisher] ERROR: failed to open {self.HUNGER_PORT_NAME}")
            return False
        print(f"[MockPublisher] Port open: {self.HUNGER_PORT_NAME}")

        print(
            f"[MockPublisher] Ready — "
            f"context every {self.context_period}s, "
            f"HS every {self.hs_period}s"
        )
        return True

    def getPeriod(self) -> float:
        """Run updateModule at ~10 Hz; actual publish cadence managed internally."""
        return 0.1

    def updateModule(self) -> bool:
        now = time.time()

        # --- context:o publish ---
        if now - self._last_context_ts >= self.context_period:
            self._publish_context()
            self._last_context_ts = now
            self.episode_id += 1   # increment episode_id each publish to simulate progress
            self.chunk      += 1

        # --- hunger:o publish ---
        if now - self._last_hs_ts >= self.hs_period:
            self._publish_hs()
            self._last_hs_ts = now

        return True

    def close(self) -> bool:
        print("[MockPublisher] Closing ports …")
        self.context_port.close()
        self.hunger_port.close()
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _publish_context(self) -> None:
        """
        Mirrors ShortTermMemoryV2.publish_episode_context():
            Bottle: Int32(episode_id) | Int32(chunk) | Int8(label)
        """
        btl = yarp.Bottle()
        btl.clear()
        btl.addInt32(self.episode_id)
        btl.addInt32(self.chunk)
        btl.addInt8(self.label)
        self.context_port.write(btl)
        print(
            f"[MockPublisher] context:o  → "
            f"episode_id={self.episode_id}  chunk={self.chunk}  label={self.label}"
        )

    def _publish_hs(self) -> None:
        """
        Mirrors InteractionManagerModule.updateModule() hunger publish:
            Bottle: String(hs)   where hs ∈ {"HS1", "HS2", "HS3"}
        """
        if self.fixed_hs:
            hs = self.fixed_hs
        else:
            hs = self.HS_STATES[self.hs_index % len(self.HS_STATES)]
            self.hs_index += 1

        b = self.hunger_port.prepare()
        b.clear()
        b.addString(hs)
        self.hunger_port.write()
        print(f"[MockPublisher] hunger:o   → {hs}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_argv_for_rf(args: argparse.Namespace) -> list[str]:
    """Convert argparse namespace to a flat list suitable for YARP RF."""
    yarp_argv = [sys.argv[0]]
    for key, val in vars(args).items():
        if val is not None and val != "":
            yarp_argv += [f"--{key}", str(val)]
    return yarp_argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock publisher for STM context and IM hunger ports")
    parser.add_argument("--context-period", type=float, default=2.0,  metavar="SEC",
                        help="Seconds between context:o publishes (default: 2.0)")
    parser.add_argument("--hs-period",      type=float, default=1.0,  metavar="SEC",
                        help="Seconds between hunger:o publishes (default: 1.0)")
    parser.add_argument("--episode-id",     type=int,   default=0,    metavar="N",
                        help="Starting episode ID (default: 0)")
    parser.add_argument("--chunk",          type=int,   default=-1,   metavar="N",
                        help="Starting chunk ID (default: -1)")
    parser.add_argument("--label",          type=int,   default=0,    metavar="N",
                        help="Context label (default: 0)")
    parser.add_argument("--hs",             type=str,   default="",   metavar="HS1|HS2|HS3",
                        help="Fixed hunger state to publish (default: cycles HS1→HS2→HS3)")
    args = parser.parse_args()

    yarp.Network.init()
    if not yarp.Network.checkNetwork():
        print("[MockPublisher] ERROR: YARP network not available. Is yarpserver running?")
        sys.exit(1)

    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.configure(_build_argv_for_rf(args))

    module = MockPublisher()
    module.setName("mockPublisher")
    module.runModule(rf)

    yarp.Network.fini()


if __name__ == "__main__":
    main()
