import yaml

from src.post_process.game_controller.data_transmutation import DataTransmuter

def main(config):
    metadata, data = DataTransmuter(config).stir()
    if metadata:
        with open(config.get_paths().gameinfo, "w") as f:
            yaml.safe_dump(metadata.params.game, f)
    for section_name in data:
        paths = config.get_paths(gc_section_name=section_name)

        paths.gc_section_dir.mkdir(exist_ok=True)
        data[section_name].state_records.to_csv(paths.game_state_csv, index=False)
        data[section_name].players_collective_records.to_csv(paths.gc_raw_csv, index=False)

        paths.gc_individual_dir.mkdir(exist_ok=True)
        for key in data[section_name].players_individual_records.keys():
            player, team = key
            data[section_name].players_individual_records[key].to_csv(paths.gc_individual_dir / f"team{team}_player{player}.csv", index=False)

        data[section_name].events_records.to_csv(paths.gc_events_csv, index=False)
        data[section_name].penalties_records.to_csv(paths.gc_penalties_csv, index=False)

        with open(paths.gc_metadata_yaml, "w") as f:
            yaml.safe_dump({"name": section_name, "phase": data[section_name].phase}, f)
