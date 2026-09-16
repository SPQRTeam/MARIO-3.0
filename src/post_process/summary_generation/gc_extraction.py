import yaml

from src.post_process.game_controller.data_transmutation import DataTransmuter

def main(config):
    metadata, data = DataTransmuter(config).stir()
    if metadata:
        with open(config.gameinfo_path, "w") as f:
            yaml.safe_dump(metadata.params.game, f)
    for section_name in data:
        (config.game_dir / section_name).mkdir(exist_ok=True)
        data[section_name].state_records.to_csv(config.game_state_path(section_name), index=False)
        data[section_name].players_collective_records.to_csv(config.game_dir / section_name / "gc_collective.csv", index=False)
        (config.game_dir / section_name / "gc_individual").mkdir(exist_ok=True)
        for key in data[section_name].players_individual_records.keys():
            player, team = key
            data[section_name].players_individual_records[key].to_csv(config.game_dir / section_name / "gc_individual" / f"team{team}_player{player}.csv", index=False)
        data[section_name].events_records.to_csv(config.game_dir / section_name / "events_from_gc.csv", index=False)
        data[section_name].penalties_records.to_csv(config.game_dir / section_name / "penalties.csv", index=False)
        with open(config.game_dir / section_name / "metadata.yaml", "w") as f:
            yaml.safe_dump({"name": section_name, "phase": data[section_name].phase}, f)
