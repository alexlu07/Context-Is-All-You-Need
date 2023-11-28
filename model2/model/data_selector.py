import pandas as pd
import numpy as np
from fastprogress import progress_bar

class DataSelector:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)

    def select_data(self, length=5, ratio=None, filter=None, seed=12345678):
        if ratio: 
            ratio = pd.Series(ratio)
            assert set(ratio.index) == set(pd.unique(self.df['source']))

        print("Filtering...")

        df = self.df.groupby("conversation").filter(lambda x: x["text"].count() >= length)
        df = filter(df) if filter else df

        print("Calculating counts...")

        source_groups = df.groupby("source")

        group_counts = df.groupby("source").apply(lambda x: sum(x.groupby("conversation")["text"].count()+1-length))
        counts = ((group_counts / ratio).min() * ratio).astype(int)

        print("Grabbing data...")

        rng = np.random.default_rng(seed)

        final_data = []
        for source, frame in progress_bar(source_groups):
            convos = frame.groupby("conversation")
            convos_lens = convos["text"].count()

            choices = [(c, i) for c, l in convos_lens.items() for i in range(l+1-length)]

            print(len(choices))
            selected = rng.choice(len(choices), counts[source], replace=False, )

            for idx in selected:
                c, i = choices[idx]
                conv = convos.get_group(c)
                final_data.append([conv.iloc[i:i+length-1]["text"].values, c, source] + conv.iloc[i+length-1].loc["0":"27"].values.tolist())

        return pd.DataFrame(final_data, columns=["text", "conversation", "source"] + [str(x) for x in range(28)])