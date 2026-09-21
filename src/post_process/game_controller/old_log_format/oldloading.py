from .old_tcm_loading import iterate_tcm_items
from .old_gc_loading import iterate_oldgc_items

from munch import RecursiveMunch

import src.post_process.game_controller.gc_utils as gc_utils

"""
This is the entry point of this module.

The biggest difference b/w new and old loading is that penalties and set plays are noted by number rather than by name.
This is b/c the relaitionship b/w penalties and numbers changed frequently, and this doesn't know which rules were in effect at the time of each log.
"""


def iterator_that_ends_on_none_and_supports_timestamp_offsetting(orig_iterator, offset_timestamp=None):
    while True:
        try:
            the_next = next(orig_iterator)
            if offset_timestamp is not None:
                the_next["timestamp"] = gc_utils.timestamp_add(the_next["timestamp"], offset_timestamp)
            yield the_next
        except StopIteration:
            yield None

def entry_is_internal(entry):
    if entry["__type__"] == "action":
        if entry["action"]["type"] == "stateChange":
            return True
    # metadata records are incomplete, don't send them by themselves
    if entry["__type__"] == "metadata_from_gc":
        return True
    if entry["__type__"] == "metadata_from_tcm":
        return True
    return False

def should_do_tcm(next_gc, next_tcm):
    if next_tcm == None:
        return False
    if gc_utils.timestamp_to_float(next_tcm["timestamp"]) > gc_utils.timestamp_to_float(next_gc["timestamp"]):
        return False
    return True

# also does SIDE-EFFECT on gc_metadata
def construct_metadata(gc_metadata, tcm_metadata):
    # this is only done for the first TCM section, so assume the first team there is home
    # TODO is there a better way to do this?
    gc_metadata.params.game.teams.home.number = tcm_metadata.teams.first.number
    gc_metadata.params.game.teams.away.number = tcm_metadata.teams.second.number
    return gc_metadata

def iterate_old_gc_and_tcm_logs(oldgc_path, tcmlog_paths):
    gc_iter = iterator_that_ends_on_none_and_supports_timestamp_offsetting(iterate_oldgc_items(oldgc_path))
    next_gc = next(gc_iter)
    tcm_iter = None
    next_tcm = None
    metadata = None
    received_metadata_from_gc = False
    received_metadata_from_tcm = False
    pending_state_changes = []
    last_game_state_record = None

    while True:
        if should_do_tcm(next_gc, next_tcm):
            entry = next_tcm["entry"]
            if entry["__type__"] == "metadata_from_tcm":
                # each log gives metadata, but we only want to handle the first one, as the others will be repeats and we only want to yield metadata once
                if not received_metadata_from_tcm:
                    assert received_metadata_from_gc
                    metadata = construct_metadata(metadata, entry["metadata"])
                    yield RecursiveMunch.fromDict({
                        "timestamp": next_tcm.timestamp,
                        "entry": {
                            "__type__": "metadata",
                            **metadata,
                        }
                    })
                    received_metadata_from_tcm = True
            elif entry["__type__"] == "game_state":
                # add home/away info
                if entry.teams.first.number == metadata.params.game.teams.home.number:
                    entry.teams.home = entry.teams.first
                    entry.teams.away = entry.teams.second
                else:
                    entry.teams.home = entry.teams.second
                    entry.teams.away = entry.teams.first
                del entry.teams.first
                del entry.teams.second
                if entry.kickingTeamNumber == metadata.params.game.teams.home.number:
                    entry.kickingSide = "home"
                else:
                    entry.kickingSide = "away"
                del entry.kickingTeamNumber
                # discharge all pending state changed from the GC, so we actually get initial and finished
                original_state = entry.state
                while pending_state_changes:
                    entry.state = pending_state_changes.pop(0)
                    yield RecursiveMunch.fromDict({
                        "timestamp": next_tcm.timestamp,
                        "entry": entry
                    })
                entry.state = original_state
                last_game_state_record = next_tcm
            if not entry_is_internal(entry):
                yield next_tcm
            next_tcm = next(tcm_iter)
        # otherwise, should do gc
        elif next_gc == None:  # note that if next_tcm is None then we end up here regardless
            # discharge any remaining state changes (the final finish, in particular)
            while pending_state_changes:
                last_game_state_record.entry.state = pending_state_changes.pop(0)
                yield last_game_state_record
            if tcmlog_paths:
                print(f"Finishing iteration with {len(tcmlog_paths)} tcm files left :(")
            break  # done!
        else:
            timestamp = next_gc["timestamp"]
            entry = next_gc["entry"]
            # check for entering/exiting a section
            if entry["__type__"] == "action":
                if entry["action"]["type"] == "stateChange":
                    new_state = entry["action"]["args"]["state"].lower()
                    if new_state in {"ready", "set", "play", "playing"} and tcm_iter is None:
                        # we are entering a section
                        tcm_iter = iterator_that_ends_on_none_and_supports_timestamp_offsetting(iterate_tcm_items(tcmlog_paths.pop(0)), offset_timestamp=timestamp)
                        next_tcm = next(tcm_iter)
                    elif new_state in {"initial", "finish", "finished"} and tcm_iter is not None:
                        # we are exiting a section
                        if next_tcm != None:
                            print("Leaving a section with some tcm :(")  # looks like it shouldn't happen
                        tcm_iter = None
                        next_tcm = None
                    # mark the state change so it can be reported properly at the next occasion
                    # (the TCM logs are only while playing, so initial and finished are not reported)
                    # (delay this to use an up-to-date game state)
                    pending_state_changes.append(new_state)
            # also check for metadata
            elif entry["__type__"] == "metadata_from_gc":
                assert not received_metadata_from_gc, "Expect one GC metadata"
                metadata = entry["metadata"]
                received_metadata_from_gc = True
                # but don't send them yet, they're still incomplete
            # anyway...
            if not entry_is_internal(entry):
                yield next_gc
            next_gc = next(gc_iter)

last_secs = -1
for x in iterate_old_gc_and_tcm_logs(
    "/home/francesco/PhD/src/robocup 2024 challenge/GC Logs/RoboCup2022Logs/FieldA/logs/log_2022-07-12_17-00-26-704.txt",
    [
        "/home/francesco/PhD/src/robocup 2024 challenge/GC Logs/RoboCup2022Logs/FieldA/logs_teamcomm/teamcomm_2022-07-12_17-07-22-998_7v7_B-Human_Bembelbots_1stHalf.log",
        "/home/francesco/PhD/src/robocup 2024 challenge/GC Logs/RoboCup2022Logs/FieldA/logs_teamcomm/teamcomm_2022-07-12_17-29-10-883_7v7_B-Human_Bembelbots_2ndHalf.log",
    ]
):
    assert x["timestamp"]["secs"] >= last_secs, f'{x["timestamp"]["secs"]} > {last_secs}'
    last_secs = x["timestamp"]["secs"]
