import tqdm

class Neotrack:
    def __init__(self, color):
        self.color = color
        self.indices = []

    def __len__(self):
        return len(self.indices)

    def append(self, i):
        self.indices.append(i)

class _ColorProcessor:
    def __init__(self, df, config):
        self.input_df = df.copy()
        self.output_df = df.copy()
        self.next_id = df["id"].max() + 1
        self.config = config

    def _handle_track(self, input_track_id):
        track_selector = self.input_df["id"]==input_track_id
        track_indices = self.input_df[track_selector].index
        track_len = len(track_indices)
        prev_color = None
        neotracks = []
        for x in range(track_len):
            this_index = track_indices[x]
            # xmin included, xmax excluded, as per normal python convention
            xmin = max(x-self.config.trackcolor_window_length//2, 0)
            xmax = min(x+self.config.trackcolor_window_length//2, track_len)
            window_indices = track_indices[xmin:xmax]
            window = self.input_df.loc[window_indices]
            color_counts = window.color.value_counts(sort=True, ascending=False)
            max_count = color_counts.iloc[0]
            max_color = color_counts.index[0]
            # TEMP MISURA TEMPORANEA: verde e blu sono uguali
            # if max_color == "gray" or max_color == "grey":
            #     max_color = "green"
            # if max_color == "blue":
            #     max_color = "green"
            # if max_color == "yellow":
            #     max_color = "green"

            if prev_color is None or prev_color != max_color:
                # if input_track_id == 5: print("ZAN ZAN ZAN")
                neotracks.append(Neotrack(max_color))
            neotracks[-1].append(this_index)
            prev_color = max_color

        # roba che accumula neotracce troppo corte in una singola neotraccia ambigua
        i = 0
        # if input_track_id == 5: print("Early neotracks:", len(neotracks))
        while i < len(neotracks):
            if len(neotracks[i]) < self.config.neotrack_unambiguous_length:
                # if input_track_id == 5: print(self.next_id, "ambiguous!!")
                amb_indices = []
                amb_colors = set()
                while i < len(neotracks) and len(neotracks[i]) < self.config.neotrack_unambiguous_length:
                    # if input_track_id == 5: print("    len", len(neotracks[i]), neotracks[i].color)
                    amb_indices.extend(neotracks[i].indices)
                    amb_colors.add(neotracks[i].color)
                    i += 1
                self._writeout(amb_indices, amb_colors)
                # i has already been incremented at the end of the while
            else:
                # if input_track_id == 5: print(self.next_id, "unambiguous w/ len", len(neotracks[i]), neotracks[i].color)
                self._writeout(neotracks[i].indices, {neotracks[i].color})
                i += 1

    def _writeout(self, indices, color):
        self.output_df.loc[indices, ("id", "color")] = (self.next_id, str(color))
        self.next_id += 1

    def go(self, use_progressbar=True):
        iterids = self.input_df.id.unique()
        if use_progressbar:
            iterids = tqdm.tqdm(iterids)
        for track_id in iterids:
            self._handle_track(track_id)
        return self.output_df

def go(df, config):
    return _ColorProcessor(df, config).go()
