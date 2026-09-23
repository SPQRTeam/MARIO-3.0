import yaml
import os
from tqdm import tqdm
import munch
import src.post_process.game_controller.messages as gc_messages

# CLoader requires libyaml but it's FAAAAAAST
class GCLoader(yaml.CLoader):
    pass

def basic_dict_constructor_factory(the_type):
    def f(loader, node):
        if isinstance(node, yaml.MappingNode):
            return dict(
                __type__=the_type,
                **loader.construct_mapping(node, deep=True),
            )
        elif isinstance(node, yaml.SequenceNode):
            return dict(
                __type__=the_type,
                content=loader.construct_sequence(node, deep=True),
            )
        else:
            return dict(
                __type__=the_type,
                value=loader.construct_scalar(node, deep=True),
            )
    return f

# metadata get nothing special
GCLoader.add_constructor("!metadata", basic_dict_constructor_factory("metadata"))

# monitor requests get nothing special - we don't really care about external monitors
GCLoader.add_constructor("!monitorRequest", basic_dict_constructor_factory("monitor_request"))

# this is important
def gc_return_constructor(loader, node):
    try:
        yaml_stuff = loader.construct_mapping(node, deep=True)
        decoded_data = gc_messages.GCReturnMessage_v4.from_base64(yaml_stuff["data"])
        yaml_stuff_without_data = {k:d for k,d in yaml_stuff.items() if k!="data"}
        return dict(
            __type__="status_message",
            **yaml_stuff_without_data,
            **decoded_data,
        )
    except gc_messages.SkippableGCMessage:
        return "__invalid__"
GCLoader.add_constructor("!statusMessage", gc_return_constructor)

# actions don't have binary data, and a check in the following code confirms that we don't really care about them either
GCLoader.add_constructor("!action", basic_dict_constructor_factory("action"))

# the game state doesn't need special handling either, yay!
GCLoader.add_constructor("!gameState", basic_dict_constructor_factory("game_state"))

# i'm not messing with timer states until i need them
GCLoader.add_constructor("!started", basic_dict_constructor_factory("started"))
GCLoader.add_constructor("!expire", basic_dict_constructor_factory("expire"))

# team messages are whatever, they're packed in a team-specific format so we can't read them anyway
GCLoader.add_constructor("!teamMessage", basic_dict_constructor_factory("team_message"))


def iterate_yaml_list_items(yaml_path):
    with open(yaml_path) as f:
        assert f.read(2) == "- ", "This assumes the top-level object of the yaml file is a list, won't work any other way"

    total_bytes = os.path.getsize(yaml_path)
    lines_of_current_item = []
    last_record_was_action = False

    with tqdm(total=total_bytes, unit='B', unit_scale=True, unit_divisor=1024, desc="Parsing YAML") as pbar:
        with open(yaml_path, "rb") as f:  # read in binary mode just for counting the progress in bytes
            while True:
                raw_line = f.readline()
                pbar.update(len(raw_line))
                # then immediately decode the text and forget about the bytes
                line = raw_line.decode('utf-8')

                if line.startswith("- ") or not raw_line:
                    # found beginning of new item or reached end of file, send current item for processing, then...
                    if lines_of_current_item:
                        item = munch.munchify(yaml.load("".join(lines_of_current_item), GCLoader)[0])
                        if item["entry"] == "end":
                            break
                        elif item["entry"] != "__invalid__":
                            # check that every action record is followed by a game_state record.
                            # if so, we can safely avoid paying attention to actions to update the game state since it's always given in full.
                            assert (not last_record_was_action) or item["entry"]["__type__"] == "game_state"
                            last_record_was_action = item["entry"]["__type__"] == "action"
                            yield item
                    # ...start fresh if there is still data
                    if raw_line:
                        lines_of_current_item.clear()
                    # ...or stop if EOF
                    else:
                        break

                # add line to buffer and wait for the end of this item
                lines_of_current_item.append(line)