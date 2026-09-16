import json
import yaml
import munch
from dataclasses import dataclass
from copy import deepcopy
from src.ml.models.openai_chat_wrapper import commentary_llm_from_config

def build_system_prompt(home, away):
    return f"""
You are a soccer sports analyst working for a university.
You must produce a summary of the most relevant events in this particular game so that it can be indexed and searched for in the future.

Here are a few facts:

=== DATA FORMAT ===
You will receive a JSON document indicating some overall statistics about the game, such as ball possession, number of passes, shots, goals, and other events.
Use them as appropriate to provide an overview of what happened in the game.
Here is a definition of all the events:
- pass: a player successfully passed the ball to a teammate
- shot: a player took a shot at the opponents' goal (whether successful or not, is not indicated)
- goal: a player successfully scored a goal
- ball_movement: any other small play of the ball, indicates ball possession
- throwIn: the team gets to play the ball from the side line because the opponents kicked it out
- goalKick: the team gets to play the ball from their goal because the opponents kicked it out
- cornerKick: the team gets to play a corner kick because the opponents kicked it out
- directFreeKick: the team gets to play the ball because the opponent committed a foul
- foul: a player broke a rule. NOT a yellow or red card.

=== TEAMS ===
The two teams playing in this game are:
- {home.name}
- {away.name}

=== STYLE ===
Write in a flat, concise, factual style.
You are writing an abstract that will represent the game in a search engine for research use.
Basic statistics and particular events should stand out in your summary, so they can be found by people using the search engine.
However, do not repeat the numeric statistics you read in the provided documents. Simply skip the numbers and go straight to your interpretation.
The numeric data will already be available to the search engine. Your summary must accompany them in a human-readable way, not replace them.
Note: the search engine is for academic use and will be used by a small amount of people in one scientific community. This is not a commercial endavor, so not engage in SEO tactics. Keep it factual and professional.
"""

def build_user_prompt(stats_string, score_string):
    return f"""
The following is the JSON data regarding the game statistics.

{stats_string}

The game resulted in the following score: {score_string}
"""

def build_user_prompt_for_phase_compare(prev_output, statsstr1, statsstr2, scorestr1, scorestr_final):
    return f"""
You have already produced the following summary of the game as a whole:

```
{prev_output}
```

Now enhance this by adding a shorter paragraph in the same style comparing the statistics of the first and the second half, possibly describing how the situation evolved during the course of the game.
Focus ONLY on this, refrain from making general statements about the game since that's already been done previously.

----------
First half
----------

First half JSON data:
{statsstr1}

Score at the end of the first half: {scorestr1}

-----------
Second half
-----------

Second half JSON data:
{statsstr2}

Final score of the game: {scorestr_final}
"""

def _dictadd(a, b):
    for bkey in b:
        if bkey in a:
            if isinstance(b[bkey], dict):
                _dictadd(a[bkey], b[bkey])
            else:
                a[bkey] += b[bkey]
        else:  #  bkey not in a
            a[bkey] = deepcopy(b[bkey])

@dataclass
class SectionData:
    phase: str
    left_team: munch.Munch
    right_team: munch.Munch
    home_team: munch.Munch
    away_team: munch.Munch
    event_stats: dict

    def add(self, other):
        assert self.phase == other.phase
        assert self.left_team == other.left_team
        assert self.right_team == other.right_team
        assert self.home_team == other.home_team
        assert self.away_team == other.away_team
        _dictadd(self.event_stats, other.event_stats)

    def get_score_string(self):
        return f"{self.home_team.name} {self.event_stats['goal'][self.home_team.name]} - {self.away_team.name} {self.event_stats['goal'][self.away_team.name]}"

    @classmethod
    def make_combined(cls, *args):
        all_event_stats = dict()
        for section in args:
            _dictadd(all_event_stats, section.event_stats)
        return cls(
            phase="combined",
            left_team=None,
            right_team=None,
            home_team=args[0].home_team,
            away_team=args[0].away_team,
            event_stats = all_event_stats,
        )

def process_single_section(mario_section_name, config):
    with open(config.game_dir / mario_section_name / "gc_section_backlink.json") as f:
        gc_section_name = json.load(f)["gc_section_name"]
    with open(config.gameinfo_path) as f:
        gameinfo = munch.munchify(yaml.safe_load(f))
    with open(config.team_mapping_lr_path(mario_section_name)) as f:
        team_mapping_lr = munch.munchify(json.load(f))
    with open(config.game_dir / mario_section_name / "sumgen_output" / "stats.json") as f:
        basic_stats = json.load(f)

    with open(config.game_dir / gc_section_name / "metadata.yaml") as f:
        phase = yaml.safe_load(f)["phase"]

    left_team = munch.Munch(
        number = team_mapping_lr.left_number,
        color = team_mapping_lr.left_color_name,
        name = team_mapping_lr.left_full_name,
    )
    right_team = munch.Munch(
        number = team_mapping_lr.right_number,
        color = team_mapping_lr.right_color_name,
        name = team_mapping_lr.right_full_name,
    )
    if left_team.number == gameinfo.teams.home.number:
        home_team = left_team
        away_team = right_team
    else:
        home_team = right_team
        away_team = left_team

    # limit the info we provide from the JSON and make them more explicit to maximize comprehension by the LLM
    # take just the events, the important part. Later we feed event names directly, to avoid cross-referencing.
    event_stats = basic_stats["events"]
    # make sure goals are preesent, they also indicate the score
    if "goal" not in event_stats:
        event_stats["goal"] = {"total": 0, "home": 0, "away": 0}
    # change penalizedRobot to something more immediately understandable
    if "penalizedRobot" in event_stats:
        event_stats["foul"] = event_stats.pop("penalizedRobot")
    for _, eventdata in event_stats.items():
        # remove the total, it's not really interesting
        del eventdata["total"]
        # use team names directly, so it doesn't have to cross-reference who is "home" or "away"
        eventdata[home_team.name] = eventdata.pop("home")
        eventdata[away_team.name] = eventdata.pop("away")

    return SectionData(
        phase=phase,
        left_team=left_team,
        right_team=right_team,
        home_team=home_team,
        away_team=away_team,
        event_stats=event_stats,
    )



def main(mario_section_names, config):
    llm = commentary_llm_from_config(config.models)

    data_by_phase = dict()
    for sn in mario_section_names:
        section_data = process_single_section(sn, config)
        if section_data.phase in data_by_phase:
            data_by_phase[section_data.phase].add(section_data)
        else:
            data_by_phase[section_data.phase] = section_data

    if "firstHalf" in data_by_phase and "secondHalf" in data_by_phase:
        # big assert
        fh = data_by_phase["firstHalf"]
        sh = data_by_phase["secondHalf"]
        assert fh.home_team == sh.home_team and fh.away_team == sh.away_team and fh.left_team == sh.right_team and fh.right_team == sh.left_team
        main_data = SectionData.make_combined(fh, sh)
    elif "firstHalf" in data_by_phase:
        main_data = data_by_phase["firstHalf"]
    elif "secondHalf" in data_by_phase:
        main_data = data_by_phase["secondHalf"]
    else:
        raise ValueError("At least one between the first and second half are expected at present!")

    system_prompt = build_system_prompt(main_data.home_team, main_data.away_team)
    gen_summary = llm.llm_call(system_prompt + build_user_prompt(json.dumps(main_data.event_stats), main_data.get_score_string()))

    print(gen_summary)

    if "firstHalf" in data_by_phase and "secondHalf" in data_by_phase:
        user_prompt = build_user_prompt_for_phase_compare(
            gen_summary,
            json.dumps(data_by_phase["firstHalf"].event_stats),
            json.dumps(data_by_phase["secondHalf"].event_stats),
            data_by_phase["firstHalf"].get_score_string(),
            main_data.get_score_string(),
        )

        compare_output = llm.llm_call(system_prompt + user_prompt)
        gen_summary += "\n\n" + compare_output

        print("--------------------------------------------------------------")
        print("==============================================================")
        print("--------------------------------------------------------------")
        print(user_prompt)
        print()
        print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        print()
        print(compare_output)

    (config.game_dir / "sumgen_global").mkdir(exist_ok=True)
    with open(config.game_dir / "sumgen_global" / "summary.txt", "w") as f:
        f.write(gen_summary)
