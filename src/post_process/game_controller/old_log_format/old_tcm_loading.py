from pathlib import Path
import struct
import sys

# unfortunately, this code was developed on a system just slightly too old to use torch in python 3.12,
# even in a pixi env, so javaobj.v3 was not an option.
# An hypothetical future migration entails:
# - using get_field rather than direct access in all pobjs
# - simplifying the handling of bytes in the case common.net.GameControlReturnDataPackage
#   (v3 directly gives a bytes object, no need for passing through struct)
import javaobj.v2 as javaobj
from munch import RecursiveMunch

sys.path.insert(0, str(Path(__file__).parents[4]))
import src.post_process.game_controller.messages as gc_messages

STATE_NAMES = [
    "initial",   # 0
    "ready",     # 1
    "set",       # 2
    "playing",   # 3
    "finished",  # 4
]

def blockdata_from_bytes(the_bytes):
    data = struct.unpack('> q ?', the_bytes)
    return {
        "time": data[0],
        "is_timeout_event": data[1],
        # from what i gather, we don't care about timeout events at all
        # These are moments where the GC received no message for a while, not a "gameplay" timeout.
        # In these moments, the GC resets the section and dumps the log file, so I don't think it matter for us.
    }

def _player(pobj):
    return {
        "penalty": pobj.penalty,
    }
def _team(pobj):
    return {
        "number": pobj.teamNumber,
        "score": pobj.score,
        "players": [_player(p) for p in pobj.player]
    }
def game_state_from_pobj(pobj):
    return {
        "phase": "firstHalf" if pobj.firstHalf else "secondHalf",
        "state": STATE_NAMES[pobj.gameState],
        "setPlay": pobj.setPlay,
        "kickingTeamNumber": pobj.kickingTeam,
        "teams": {
            "first": _team(pobj.team[0]),
            "second": _team(pobj.team[1]),
        },
    }


def iterate_tcm_items(tcmlog_path):
    with open(tcmlog_path, "rb") as fd:
        all_pobjs = javaobj.load(fd)

    time = None
    first_game_control_data = True

    for pobj in all_pobjs:
        try:
            typename = pobj.classdesc.name
        except AttributeError:
            typename = "BlockData"

        if typename == "BlockData":
            bd = blockdata_from_bytes(pobj.data)
            time = bd["time"]
        elif typename == "common.net.GameControlReturnDataPackage":
            # this little passage is necessary b/c javaobj.v2 has bytes as lists of signed integers
            b = struct.pack(f"{len(pobj.message.data)}b", *pobj.message.data)
            yield RecursiveMunch.fromDict({
                "timestamp": {
                    "secs": time // 1000,
                    "nanos": 0,  # Exclude nanoseconds because the gc log doesn't have them
                },
                "entry": {
                    "__type__": "status_message",
                    "host": pobj.host,
                    **gc_messages.GCReturnMessage_v4.from_bytes(b),
                },
            })
        elif typename == "data.AdvancedData":
            if first_game_control_data:
                yield RecursiveMunch.fromDict({
                    "timestamp": {
                        "secs": time // 1000,
                        "nanos": 0,  # Exclude nanoseconds because the gc log doesn't have them
                    },
                    "entry": {
                        "__type__": "metadata_from_tcm",
                        "metadata": {
                            "teams": {
                                "first": {
                                    "number": pobj.team[0].teamNumber,
                                },
                                "second": {
                                    "number": pobj.team[1].teamNumber,
                                }
                            }
                        }
                    },
                })
                first_game_control_data = False
                # then also yield the normal game state next
            yield RecursiveMunch.fromDict({
                "timestamp": {
                    "secs": time // 1000,
                    "nanos": 0,  # Exclude nanoseconds because the gc log doesn't have them
                },
                "entry": {
                    "__type__": "game_state",
                    **game_state_from_pobj(pobj),
                },
            })
        elif typename == "common.net.SPLStandardMessagePackage":
            # In this case, we don't really care. The standard part is the same as GameControlReturnData,
            # and the arbitrary payload is team-specific so we can't read it.
            # In theory, the standard part could be used to get extra "GameControlReturn"s,
            # but at present, it's not urgent.
            pass
        else:
            raise TypeError(f"Unknown TCM-log type: {typename}")

# states = set()
# for x in iterate_tcm_items("/home/francesco/PhD/src/robocup 2024 challenge/GC Logs/RoboCup2022Logs/FieldB/logs_teamcomm/teamcomm_2022-07-13_10-46-07-863_SABANA Herons_HULKs_1stHalf.log"):
#     if x["entry"]["__type__"] == "game_state":
#         states.add(x["entry"]["state"])
# print(states)

