import re
from datetime import datetime
from munch import RecursiveMunch

GLOBAL_PATTERN = re.compile(r"(\d{4}) \. (\d{1,2}) \. (\d{1,2}) - (\d{2}) \. (\d{2}) \. (\d{2}) : (.*)", re.VERBOSE)
UNDO_PATTERN = re.compile(r".*? undo (\d+) states to (.*)", re.IGNORECASE)
VERSUS_PATTERN = re.compile(r"(.*?) \((.*?)\) vs (.*?) \((.*?)\)")
GOAL_PATTERN = re.compile(r"Goal for (.*)")

GAME_STATES = {"initial", "ready", "set", "playing", "finished"}
PENALTIES = {"Inactive Player", "Request for Pickup", "Illegal Position", "Leaving the Field", "Player Pushing", "Illegal Motion in Set", "Local Game Stuck"}
FREE_KICKS = {"Goal Kick", "Pushing Free Kick", "Corner Kick", "Kick In"}

# SIDE EFFECT
def _apply_undo(lines):
    i = 0
    while i < len(lines):
        m = re.match(UNDO_PATTERN, lines[i])
        if m:
            undo_num = int(m.group(1))
            del lines[i-undo_num:i+1]  # delete the undone lines as well as the undo log itself
            i -= undo_num + 1
        i += 1

def iterate_oldgc_items(oldgc_path):

    with open(oldgc_path) as f:
        lines = f.readlines()

    _apply_undo(lines)

    initial_time = None
    encountered_versus = False
    home_color = None
    away_color = None

    def _color_to_side(color):
        if color == home_color:
            return "home"
        if color == away_color:
            return "away"
        raise ValueError(f"Team color mismatch in goal: {color}/{home_color}/{away_color}")

    for line in lines:
        m = re.match(GLOBAL_PATTERN, line)
        year, month, day, hour, minute, second, message = m.groups()
        time = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        if initial_time is None:
            initial_time = time
        timestamp = int((time - initial_time).total_seconds())
        message = message.strip()

        m = re.match(VERSUS_PATTERN, message)
        if m:
            encountered_versus = True
            home_color = m.group(2)
            away_color = m.group(4)
            yield RecursiveMunch.fromDict({
                "timestamp": {
                    "secs": timestamp,
                    "nanos": 0,
                },
                "entry": {
                    "__type__": "metadata_from_gc",
                    "metadata": {
                        "params": {
                            "game": {
                                "teams": {
                                    "home": {
                                        "fieldPlayerColor": home_color,
                                        "goalkeeperColor": home_color,
                                    },
                                    "away": {
                                        "fieldPlayerColor": away_color,
                                        "goalkeeperColor": away_color,
                                    },
                                }
                            }
                        }
                    }
                }
            })
            continue

        m = re.match(GOAL_PATTERN, message)
        if m:
            scoring_color = m.group(1)
            yield RecursiveMunch.fromDict({
                "timestamp": {
                    "secs": timestamp,
                    "nanos": 0,
                },
                "entry": {
                    "__type__": "action",
                    "source": "user",
                    "action": {
                        "type": "goal",
                        "args": {
                            "side": _color_to_side(scoring_color)
                        }
                    }
                }
            })
            continue

        penalty_found = False
        for penalty_name in PENALTIES:
            if message.lower().startswith(penalty_name.lower()):
                penalized_color, penalized_number = message[len(penalty_name+" "):].split()
                yield RecursiveMunch.fromDict({
                    "timestamp": {
                        "secs": timestamp,
                        "nanos": 0,
                    },
                    "entry": {
                        "__type__": "action",
                        "source": "user",
                        "action": {
                            "type": "penalize",
                            "args": {
                                "side": _color_to_side(penalized_color),
                                "player": int(penalized_number),
                                "call": penalty_name,
                            }
                        }
                    }
                })
                penalty_found = True
                break
        if penalty_found:
            continue

        kick_found = False
        for kick_name in FREE_KICKS:
            if message.lower().startswith(kick_name.lower()):
                rest_of_message = message[len(kick_name+" "):]
                if rest_of_message.startswith("for"):
                    kicking_color = rest_of_message[len("for "):]
                    yield RecursiveMunch.fromDict({
                        "timestamp": {
                            "secs": timestamp,
                            "nanos": 0,
                        },
                        "entry": {
                            "__type__": "action",
                            "source": "user",
                            "action": {
                                "type": "startSetPlay",
                                "args": {
                                    "side": _color_to_side(kicking_color),
                                    "setPlay": kick_name,
                                }
                            }
                        }
                    })
                    kick_found = True
                    break
                elif rest_of_message.lower() == "complete":
                    # no utility for this event yet
                    kick_found = True
                    break
        if kick_found:
            continue

        if message.lower().startswith("timeout"):
            timeout_color = message[len("Timeout "):]
            yield RecursiveMunch.fromDict({
                "timestamp": {
                    "secs": timestamp,
                    "nanos": 0,
                },
                "entry": {
                    "__type__": "action",
                    "source": "user",
                    "action": {
                        "type": "timeout",
                        "args": {
                            "side": _color_to_side(timeout_color)
                        }
                    }
                }
            })
            continue

        if message.lower().startswith("global game stuck"):
            yield RecursiveMunch.fromDict({
                "timestamp": {
                    "secs": timestamp,
                    "nanos": 0,
                },
                "entry": {
                    "__type__": "action",
                    "source": "user",
                    "action": {
                        "type": "globalGameStuck",
                        "args": None
                    }
                }
            })
            continue

        if message.lower() in GAME_STATES:
            yield RecursiveMunch.fromDict({
                "timestamp": {
                    "secs": timestamp,
                    "nanos": 0,
                },
                "entry": {
                    "__type__": "action",
                    "source": "user",
                    "action": {
                        "type": "stateChange",
                        "args": {
                            "state": message
                        }
                    }
                }
            })
            continue


        if message.lower() == "2nd half":
            # irrelevant, the TCM already has GameControlData
            continue

        if message.lower().startswith("unpenalised"):
            # no utility for this event yet
            continue

        if message.lower().startswith("end of timeout"):
            # no utility for this event yet
            continue

        if message.lower().startswith("message budget exceeded"):
            # no utility for this event yet
            continue

        if message.lower().startswith("increase game clock"):
            # no utility for this event yet
            continue

        if message.lower() == "shutdown gamecontroller":
            # ok
            continue

        if message.lower().startswith("testmode"):
            # this is a thing that can appear at random it seems
            continue

        # if encountered_versus:
        #     raise ValueError(f"Unknown message: {message}")
