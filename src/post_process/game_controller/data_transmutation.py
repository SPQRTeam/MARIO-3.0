import munch
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
import re

from src.post_process.game_controller.loading import iterate_yaml_list_items
import src.post_process.game_controller.gc_utils as gc_utils

INACTIVE_STATES = {"initial", "finished"}
TEAM_KEYS = {
    "score": None,
    "players": {
        "penalty": None,
        "cautions": None,
    },
}
KEYS_TO_KEEP = {
    "stopped": None,
    "phase": None,
    "state": None,
    "setPlay": None,
    "kickingSide": None,
    "teams": {
        "home": TEAM_KEYS,
        "away": TEAM_KEYS,
    },
    "primaryTimer": {
        "remaining": None,
    }
}

@dataclass
class Benchwarmer:
    player_num: int
    team_num: int
    homeness: str

PATTERN = re.compile(r"teams_(home|away)_players_([0-9]+)_penalty")

class Section:
    def __init__(self, number, start_timestamp, config):
        self.number = number
        self.start_timestamp = start_timestamp
        self.phase = set()
        self.state_records = []
        self.players_collective_records = []
        self.players_individual_records = defaultdict(list)
        self.events_records = []
        self.penalties_records = []
        self.config = config

    @property
    def name(self):
        return self.config.dir_names.gc_section_prefix + str(self.number)

    @staticmethod
    def conv(df):
        if df.empty:
            return df
        else:
            return df.astype({"gctime": int})

    def finalize(self):
        self.state_records = self.conv(pd.DataFrame(self.state_records))
        self.players_collective_records = self.conv(pd.DataFrame(self.players_collective_records))
        for key in self.players_individual_records.keys():
            self.players_individual_records[key] = self.conv(pd.DataFrame(self.players_individual_records[key]))
        self.events_records = self.conv(pd.DataFrame(self.events_records))
        self.penalties_records = self.conv(pd.DataFrame(self.penalties_records))
        if len(self.phase) == 1:
            self.phase = self.phase.pop()
        else:
            print(f"Section {self.name} doesn't have a definite phase, this log must be funky!")
            print(f"The detected phases are: {self.phase}")
            self.phase = None




"""
This is the entry point.
"""
class DataTransmuter:
    def __init__(self, config):
        self.config = config
        self.current_section = None
        self.next_section_number = 0
        self.results = defaultdict(dict)
        self.metadata = None
        self.gameinfo = None
        self.homeness = dict()

    # eliminate players outside the game
    def prune_benchwarmers(self):
        # find them
        benchwarmers = []
        for key, _ in self.current_section.state_records.items():
            m = re.match(PATTERN, key)
            if m and (self.current_section.state_records[key] == "substitute").all():
                h = m.group(1)
                n = int(m.group(2))
                benchwarmers.append(Benchwarmer(n, self.gameinfo.teams[h].number, h))

        # delete from state
        to_drop = tuple(f"teams_{b.homeness}_players_{b.player_num}" for b in benchwarmers)
        for key, _ in self.current_section.state_records.items():
            if key.startswith(to_drop):
                self.current_section.state_records = self.current_section.state_records.drop(key, axis=1)

        # delete from collective players
        to_drop = [(b.player_num, b.team_num) for b in benchwarmers]
        df = self.current_section.players_collective_records  # only to shorten the name in the upcoming op
        self.current_section.players_collective_records = df[~df.set_index(['player', 'team']).index.isin(to_drop)]

        # delete from individual players
        to_drop = {(b.player_num, b.team_num) for b in benchwarmers}
        for pt in to_drop:
            self.current_section.players_individual_records.pop(pt, None)

    def on_section_start(self, item):
        # it's important to reinitialize and NOT clear in-place, or the results may be affected!
        self.current_section = Section(self.next_section_number, item.timestamp, self.config)
        self.next_section_number += 1

    def on_section_end(self):
        self.current_section.finalize()
        self.prune_benchwarmers()
        self.results[self.current_section.name] = self.current_section
        # it's important to reinitialize and NOT clear in-place, or the results may be affected!
        self.current_section = None

    def on_metadata(self, item):
        assert not self.metadata, "Expect one metadata per game"
        self.metadata = item.entry
        self.gameinfo = item.entry.params.game
        self.homeness[self.gameinfo.teams.home.number] = "home"
        self.homeness[self.gameinfo.teams.away.number] = "away"

    def on_game_state(self, item):
        # NOTE: if secsremaining is needed again, it can be found here, under the key primaryTimer->remaining->[0].
        #       maybe a constructor for !started will be required?
        csvable = munch.Munch(
            gctime=gc_utils.timestamp_to_float(gc_utils.timestamp_diff(item.timestamp, self.current_section.start_timestamp)),
        )
        filtered_state = gc_utils.filter_dict_recursive(item.entry, KEYS_TO_KEEP)
        filtered_state = gc_utils.flatten_dict_recursive(filtered_state)
        csvable.update(filtered_state)
        self.current_section.state_records.append(csvable)
        self.current_section.phase.add(item.entry.phase)

    def on_status_message(self, item):
        # get the game state as the latest one encountered in this section (the log is sequential)
        game_state = self.current_section.state_records[-1]
        my_homeness = self.homeness[item.entry.team_num]
        # can't automatize so much because of legacy column names, luckily these rows are not as big as the game state's
        individual_csvable = munch.Munch(
            gctime=gc_utils.timestamp_to_float(gc_utils.timestamp_diff(item.timestamp, self.current_section.start_timestamp)),
            playing=game_state.state == "playing",
            x=item.entry.pose.x,
            y=item.entry.pose.y,
            theta=item.entry.pose.theta,
            ballx=item.entry.ball.x,
            bally=item.entry.ball.y,
            ballage=item.entry.ball_age,
            fallen=item.entry.fallen,
            penalized=game_state[f"teams_{my_homeness}_players_{item.entry.player_num}_penalty"] != "noPenalty",
            secsremaining=game_state.primaryTimer_remaining_0,
        )
        self.current_section.players_individual_records[(item.entry.player_num, item.entry.team_num)].append(individual_csvable)
        full_csvable = munch.Munch(
            player=item.entry.player_num,
            team=item.entry.team_num,
        )
        full_csvable.update(individual_csvable)
        self.current_section.players_collective_records.append(full_csvable)

    def on_action(self, item):
        # strutture che voglio: eventi (tutti ma dati limitati), penalty (include anche il tipo), ...
        action = item.entry.action
        if action.type == "goal":
            self.current_section.events_records.append(
                munch.Munch(
                    gctime=gc_utils.timestamp_to_float(gc_utils.timestamp_diff(item.timestamp, self.current_section.start_timestamp)),
                    event="goal",
                    team=self.gameinfo.teams[action.args.side].number,
                    player="",
                )
            )
        elif action.type == "penalize":
            self.current_section.events_records.append(
                munch.Munch(
                    gctime=gc_utils.timestamp_to_float(gc_utils.timestamp_diff(item.timestamp, self.current_section.start_timestamp)),
                    event="penalizedRobot",
                    team=self.gameinfo.teams[action.args.side].number,
                    player=action.args.player,
                )
            )
            self.current_section.penalties_records.append(
                munch.Munch(
                    gctime=gc_utils.timestamp_to_float(gc_utils.timestamp_diff(item.timestamp, self.current_section.start_timestamp)),
                    penalty=action.args.call,
                    team=self.gameinfo.teams[action.args.side].number,
                    player=action.args.player,
                    # TODO mettere la duration sarebbe carino, anche se al momento non ho in mente un'utilità pratica
                )
            )
        elif action.type == "startSetPlay":
            self.current_section.events_records.append(
                munch.Munch(
                    gctime=gc_utils.timestamp_to_float(gc_utils.timestamp_diff(item.timestamp, self.current_section.start_timestamp)),
                    event=action.args.setPlay,
                    team=self.gameinfo.teams[action.args.side].number,
                    player="",
                )
            )
        # types that exist but I don't want in the report for the LLM: unpenalize, finishHalf, waitForSetPlay, freeSetPlay, finishSetPlay, teamMessage, substitute, stopPlay
        # some of them are interesting for other reasons, they're just not "events"



    def stir(self):
        if self.results:
            raise ValueError("Can only be used once.")
        for item in iterate_yaml_list_items(self.config.get_paths().gc_log):
            itemtype =  item["entry"]["__type__"]

            if itemtype == "metadata":
                self.on_metadata(item)

            if itemtype == "game_state":
                # section start
                if item.entry.state not in INACTIVE_STATES and self.current_section is None:
                    self.on_section_start(item)
                # section end
                elif item.entry.state in INACTIVE_STATES and self.current_section is not None:
                    self.on_section_end()

                # process section
                if self.current_section is not None:
                    self.on_game_state(item)

            elif itemtype == "status_message":
                # only work if inside a section
                if self.current_section is not None:
                    self.on_status_message(item)

            elif itemtype == "action":
                if self.current_section is not None:
                    self.on_action(item)
        return self.metadata, self.results
