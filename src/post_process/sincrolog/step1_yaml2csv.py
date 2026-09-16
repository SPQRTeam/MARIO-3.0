import yaml

from src.post_process.game_controller.data_transmutation import DataTransmuter

def main(config):
    metadata, data = DataTransmuter(config).stir()

    assert metadata
    with open(config.gameinfo_path, "w") as f:
        yaml.safe_dump(metadata.params.game, f)

    for section_name in data:
        section_dir = config.game_dir / section_name
        section_dir.mkdir(exist_ok=True)
        data[section_name].players_collective_records.to_csv(section_dir / config.game_controller.gc_csv_fname, index=False)









# keeping the previous version stashed and inactive below for now,
# b/c i don't know if the new DataTransmuter can handle all the old versions the hacked TCM could.
# main_legacy requires my hacked TCM:
# https://github.com/torchipeppo/RoboCupSPLTeamCommunicationMonitor/tree/for-the-csv-2026

from contextlib import contextmanager
import subprocess
import os
import re

# this is available in Python 3.11, but the env is all set up with 3.10 already, so...
# note if this legagy code is ever picked up: the env is 3.11 now!
@contextmanager
def chdir(path):
    old_cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(old_cwd)

def main_legacy(config):

    with chdir(config.tcm_dir):
        cmdlist = ["java", "-jar", "TeamCommunicationMonitor.jar", "-t",  str(config.gc_log_path)]
        print("Running", " ".join(cmdlist))
        subprocess.run(cmdlist)

    for produced_csv in (config.game_dir).rglob("gc.yaml__section*.csv"):
        section_name = "gc_" + re.match(r"gc.yaml__(section_[0-9]+).csv", produced_csv.name).group(1)
        section_dir = config.game_dir / section_name
        section_dir.mkdir(exist_ok=True)
        produced_csv.rename(section_dir / config.game_controller.gc_csv_fname)
    print("Moved results to subfolders in", config.game_dir)


    # also, extract the teams info from the log and save them in a separate file, so we don't as much overhead later
    print("Extracting gameinfo (may take some time b/c gc.yaml is big)...")
    class NoTagLoader(yaml.SafeLoader):
        pass

    def ignore_unknown(loader, tag_suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        elif isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        else:
            return loader.construct_scalar(node)

    NoTagLoader.add_multi_constructor('!', ignore_unknown)

    with open(config.gc_log_path, 'r') as f:
        data = yaml.load(f, Loader=NoTagLoader)
    gameinfo = data[0]["entry"]["params"]["game"]
    with open(config.gameinfo_path, 'w') as f:
        yaml.dump(gameinfo, f)
