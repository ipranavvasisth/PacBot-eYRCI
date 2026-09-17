#!/usr/bin/env python3
"""
Task 1A: PacBot 2D Path Planning Implementation (v2)

Changes from the original version:
  1. NAVIGATION IS PRECOMPUTED ONCE AT STARTUP instead of re-running BFS
     from scratch on every single MQTT tick. We run one BFS per maze cell
     (169 total), each rooted at that cell as the "goal". Each BFS tree
     gives us, for every other cell, both the distance to the goal and
     the next hop to take toward it. After that one-time setup, picking
     a move at runtime is an O(1) dictionary lookup per candidate pellet/
     exit instead of a full graph search.
  2. Deterministic tie-breaking: when two pellets are equally close, we
     break the tie by cell coordinates instead of relying on Python set
     iteration order (which was arbitrary and could cause target
     thrashing).
  3. Standing on an uncollected pellet (distance 0) now returns None
     (wait) instead of silently falling through to the exit phase.
  4. Out-of-bounds / post-exit poses are handled safely: the lookup
     tables only ever contain valid maze cells as keys, so an
     out-of-maze pose just fails a dict lookup instead of indexing
     WALLS with a bad row/col (which could silently wrap around via
     negative indexing or crash outright). A `finished` flag stops the
     controller from doing further work once the bot has exited.
"""

from collections import deque
import json
import time

import paho.mqtt.client as mqtt

MAZE_ROWS = 13
MAZE_COLS = 13
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
POSE_TOPIC = "robot/pose"

# wall bit per side, OR'd together
WALL_N, WALL_E, WALL_S, WALL_W = 0x1, 0x2, 0x4, 0x8

WALLS = [
    [12, 6, 12, 6, 13, 4, 0, 4, 5, 6, 12, 5, 6],
    [10, 11, 10, 10, 12, 3, 8, 2, 12, 1, 1, 6, 10],
    [8, 5, 3, 8, 1, 6, 9, 2, 9, 6, 13, 2, 10],
    [10, 12, 4, 3, 12, 1, 4, 0, 6, 9, 4, 2, 10],
    [10, 10, 8, 5, 3, 13, 2, 10, 9, 6, 10, 11, 10],
    [8, 3, 9, 4, 5, 6, 8, 1, 6, 10, 9, 5, 2],
    [10, 12, 4, 1, 6, 10, 9, 6, 10, 8, 5, 5, 2],
    [8, 1, 2, 12, 3, 10, 12, 3, 9, 2, 12, 6, 10],
    [9, 6, 10, 10, 12, 1, 2, 12, 5, 1, 0, 1, 3],
    [14, 8, 1, 1, 3, 12, 1, 3, 12, 4, 2, 12, 6],
    [8, 0, 4, 7, 12, 3, 12, 6, 10, 9, 1, 2, 10],
    [10, 10, 9, 6, 10, 12, 2, 8, 1, 7, 12, 0, 2],
    [9, 1, 5, 1, 1, 3, 8, 1, 5, 5, 3, 9, 3],
]

# the 2 known exits: (row, col, facing)
EXIT_CELLS = [
    (0, 6, 'south'),
    (MAZE_ROWS - 1, 6, 'north'),
]

BOT_CMD_TOPIC = "bot/cmd"
PELLETS_TOPIC = "pellets/pose"
CMD_VEL_TOPIC = "robot/cmd_vel"

# yaw -> dr, dc, wall bit. 0=EAST, 90=NORTH, 180=WEST, 270=SOUTH
HEADING_DELTA = {
    0.0:   (0, 1, WALL_E),
    90.0:  (1, 0, WALL_N),
    180.0: (0, -1, WALL_W),
    270.0: (-1, 0, WALL_S),
}

FACING_TO_YAW = {
    'east': 0.0,
    'north': 90.0,
    'west': 180.0,
    'south': 270.0,
}


# ============================================================================
# GRAPH UTILITIES
# ============================================================================
def in_bounds(cell):
    r, c = cell
    return 0 <= r < MAZE_ROWS and 0 <= c < MAZE_COLS


def get_neighbors(cell):
    """Returns valid accessible neighbor cells from the bitmask WALLS table."""
    r, c = cell
    neighbors = []
    cell_walls = WALLS[r][c]

    for yaw, (dr, dc, wall_bit) in HEADING_DELTA.items():
        if not (cell_walls & wall_bit):
            nr, nc = r + dr, c + dc
            if in_bounds((nr, nc)):
                neighbors.append((nr, nc))
    return neighbors


def precompute_navigation():
    """
    Run one BFS per maze cell, treating that cell as the goal. This is the
    core change: instead of re-searching the graph on every tick, we build
    two lookup tables once, at startup:

      DIST[goal][cell]  -> shortest distance from `cell` to `goal`
      NEXT[goal][cell]  -> the neighbor to step into from `cell` when
                            heading toward `goal`

    Both are O(1) dict lookups at runtime. Cost of building them is
    169 BFS runs over a 169-cell graph -- trivial, and only ever paid once.

    Assumes walls are symmetric (a wall between two cells blocks movement
    in both directions), which is the standard convention for this kind
    of bitmask maze representation.
    """
    all_cells = [(r, c) for r in range(MAZE_ROWS) for c in range(MAZE_COLS)]
    dist_tables = {}
    next_tables = {}

    for goal in all_cells:
        dist = {goal: 0}
        parent = {}
        queue = deque([goal])
        while queue:
            current = queue.popleft()
            for neighbor in get_neighbors(current):
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    parent[neighbor] = current
                    queue.append(neighbor)
        dist_tables[goal] = dist
        next_tables[goal] = parent  # parent[cell] == next hop toward goal

    return dist_tables, next_tables


def get_required_yaw(from_cell, to_cell):
    """Calculates heading angle required to step from from_cell into to_cell."""
    dr = to_cell[0] - from_cell[0]
    dc = to_cell[1] - from_cell[1]

    for yaw, (hdr, hdc, _) in HEADING_DELTA.items():
        if hdr == dr and hdc == dc:
            return yaw
    return None


def turn_command_needed(current_yaw, target_yaw):
    """Calculates the minimal turn command (FRONT, LEFT, RIGHT, BACK)."""
    diff = (target_yaw - current_yaw) % 360.0
    if diff > 180.0:
        diff -= 360.0

    if diff == 0.0:
        return "FRONT"
    elif diff == 90.0:
        return "LEFT"
    elif diff == -90.0 or diff == 270.0:
        return "RIGHT"
    elif abs(diff) == 180.0:
        return "BACK"
    return "FRONT"


# ============================================================================
# YOUR ALGORITHM GOES HERE. Everything above and below is plumbing.
# ============================================================================
def choose_command(pacbot_cell, pacbot_yaw, pellets_remaining, dist_table, next_table):
    """FRONT/LEFT/RIGHT/BACK to send now, or None. pacbot_cell=(row,col),
    pacbot_yaw one of HEADING_DELTA's keys, pellets_remaining=set of
    (row,col). dist_table/next_table come from precompute_navigation()."""

    # --- 1. PELLET COLLECTION PHASE ---
    if pellets_remaining:
        candidates = []
        for pellet in pellets_remaining:
            d = dist_table.get(pellet, {}).get(pacbot_cell)
            if d is not None:
                candidates.append((d, pellet))

        if candidates:
            # Deterministic tie-break: distance first, then cell coords --
            # not left up to arbitrary set-iteration order.
            candidates.sort(key=lambda x: (x[0], x[1]))
            best_dist, best_pellet = candidates[0]

            if best_dist == 0:
                # Already standing on the nearest pellet -- wait for the
                # external pellet-collection confirmation instead of
                # moving on before it's registered.
                return None

            next_cell = next_table[best_pellet][pacbot_cell]
            req_yaw = get_required_yaw(pacbot_cell, next_cell)
            if req_yaw is not None:
                return turn_command_needed(pacbot_yaw, req_yaw)

    # --- 2. MAZE EXIT PHASE ---
    exit_candidates = []
    for exit_row, exit_col, facing in EXIT_CELLS:
        exit_cell = (exit_row, exit_col)
        d = dist_table.get(exit_cell, {}).get(pacbot_cell)
        if d is not None:
            exit_candidates.append((d, exit_cell, facing))

    if exit_candidates:
        exit_candidates.sort(key=lambda x: (x[0], x[1]))
        best_dist, exit_cell, facing = exit_candidates[0]

        if best_dist > 0:
            next_cell = next_table[exit_cell][pacbot_cell]
            req_yaw = get_required_yaw(pacbot_cell, next_cell)
            if req_yaw is not None:
                return turn_command_needed(pacbot_yaw, req_yaw)

        # Bot has reached the exit cell; align and step out of the maze.
        exit_target_yaw = FACING_TO_YAW[facing]
        return turn_command_needed(pacbot_yaw, exit_target_yaw)

    return None
# ============================================================================


def parse_pellets(payload):
    return {tuple(cell) for cell in json.loads(payload)}


def main():
    # One-time precompute -- replaces per-tick BFS.
    dist_table, next_table = precompute_navigation()

    state = {
        "running": False,
        "pellets": set(),
        "cell": (MAZE_ROWS // 2, MAZE_COLS // 2),
        "yaw": 0.0,
        "in_flight": False,
        "got_pose": False,
        "got_pellets": False,
        "finished": False,
    }

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="Controller")

    def decide_and_send():
        if (not state["running"] or state["in_flight"] or state["finished"]
                or not state["got_pose"] or not state["got_pellets"]):
            return
        print(f"[debug] pose={state['cell']} yaw={state['yaw']} pellets={state['pellets']}")
        cmd = choose_command(state["cell"], state["yaw"], set(state["pellets"]),
                              dist_table, next_table)
        if cmd is not None:
            state["in_flight"] = True
            client.publish(CMD_VEL_TOPIC, cmd)
            print(f"[controller] {state['cell']} yaw={state['yaw']} -> {cmd}, "
                  f"pellets_left={len(state['pellets'])}")

    def on_message(client, userdata, msg):
        try:
            if msg.topic == BOT_CMD_TOPIC:
                running = msg.payload.decode().startswith("1")
                was_running = state["running"]
                state["running"] = running
                if running and not was_running:
                    decide_and_send()   # kick off the reactive loop on Start
            elif msg.topic == PELLETS_TOPIC:
                state["pellets"] = parse_pellets(msg.payload.decode())
                state["got_pellets"] = True
                decide_and_send()
            elif msg.topic == POSE_TOPIC:
                data = json.loads(msg.payload.decode())
                cell = (int(data["col"]), int(data["row"]))   # wire is swapped
                state["yaw"] = float(data.get("yaw", 0.0))
                state["in_flight"] = False   # this pose is the ack for our last command

                if in_bounds(cell):
                    state["cell"] = cell
                    state["got_pose"] = True
                    decide_and_send()   # every pose/command-ack triggers the next step
                else:
                    # Bot has stepped outside the maze grid -- nothing left
                    # to plan. Stop issuing commands instead of indexing
                    # WALLS with an out-of-range row/col.
                    if not state["finished"]:
                        print(f"[controller] pose {cell} is outside the maze "
                              f"-- treating as exit complete, stopping.")
                    state["finished"] = True
        except Exception as e:
            print("[controller] mqtt parse error:", e)

    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe([(BOT_CMD_TOPIC, 0), (PELLETS_TOPIC, 0), (POSE_TOPIC, 0)])
    client.loop_start()

    print(f"[controller] ready; sending one '{CMD_VEL_TOPIC}' command at a time, "
          f"reacting to '{POSE_TOPIC}'/'{PELLETS_TOPIC}' feedback")

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
