from munch import Munch
import json
import yaml
import pandas as pd

from src.utils.team_info import NoTagLoader

def generate_homeness(teams, gameinfo):
    if teams["left_number"] == gameinfo["teams"]["home"]["number"] and teams["right_number"] == gameinfo["teams"]["away"]["number"]:
        return Munch({"left": "home", "right": "away", teams["left_number"]: "home", teams["right_number"]: "away"})
    elif teams["left_number"] == gameinfo["teams"]["away"]["number"] and teams["right_number"] == gameinfo["teams"]["home"]["number"]:
        return Munch({"left": "away", "right": "home", teams["left_number"]: "away", teams["right_number"]: "home"})
    else:
        raise RuntimeError("MARIO metadata teams and gameinfo teams don't match")

def increment_event(the_munch, event_name, home_or_away):
    if event_name not in the_munch.keys():
        the_munch[event_name] = Munch()
        the_munch[event_name].total = 0
        the_munch[event_name].home = 0
        the_munch[event_name].away = 0
    the_munch[event_name].total += 1
    the_munch[event_name][home_or_away] += 1

def main(mario_section_name, config):
    paths = config.get_paths(mario_section_name=mario_section_name)
    with open(paths.gcsec_backlink) as f:
        gc_section_name = json.load(f)["gc_section_name"]
    paths = config.get_paths(mario_section_name=mario_section_name, gc_section_name=gc_section_name)
    with open(paths.symbolic_events_jsonl) as f:
        events = [json.loads(line) for line in f.readlines()]
    with open(paths.team_mapping_lr) as f:
        teams = json.load(f)
    with open(paths.gameinfo) as f:
        gameinfo = yaml.load(f, Loader=NoTagLoader)

    homeness = generate_homeness(teams, gameinfo)

    stats = Munch()

    # cominciamo dando qualche statistica puramente numerica
    stats.team_names = Munch()
    stats.events = Munch()

    stats.team_names[homeness.left] = events[0]["left_team_name"]
    stats.team_names[homeness.right] = events[0]["right_team_name"]

    for e in events:
        increment_event(stats.events, e["event"], homeness[e["team"]])

    # facciamo la stessa cosa ma col GC
    gc_events = pd.read_csv(paths.gc_events_csv)
    stats.gc_events = Munch()
    for _, row in gc_events.iterrows():
        if row.event not in stats.events.keys():
            increment_event(stats.gc_events, row.event, homeness[row.team])

    # mergiamo il gc negli eventi, il gc ha la precedenza e sovrascrive eventuali dati di mario (in partiolare i gol)
    for gc_evname in stats.gc_events:
        stats.events[gc_evname] = stats.gc_events[gc_evname]
    del stats.gc_events

    with open(paths.sumgen_stats_json, 'w') as f:
        json.dump(stats, f, indent=4)
