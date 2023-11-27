import pandas as pd
import numpy as np
from fastprogress import progress_bar
from sklearn.model_selection import train_test_split

class DataSelector:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)

    def select_data(self, length=5, max_text=128, ratio=None, filter=None, seed=12345678):
        if ratio: 
            ratio = pd.Series(ratio)
            assert set(ratio.index) == set(pd.unique(self.df['source']))

        print("Filtering...")

        df = self.df.groupby("conversation").filter(lambda x: x["text"].count() >= length)
        df["text"] = df["text"].str.split(n=max_text).str[:max_text].str.join(" ")
        df["filter"] = 1
        if filter: df = filter(df)

        print("Calculating counts...")

        source_groups = df.groupby("source")

        group_counts = source_groups.apply(lambda x: sum(x.groupby("conversation").tail(-(length-1)).groupby("conversation")["filter"].sum()))
        counts = ((group_counts / ratio).min() * ratio).astype(int)

        print("Grabbing data...")

        rng = np.random.default_rng(seed)

        final_data = []
        for source, frame in progress_bar(source_groups):
            convos = frame.groupby("conversation")
            convos_lens = convos["filter"].count()

            choices = [(c, i) for c, l in convos_lens.items() for i in range(l+1-length) if convos.get_group(c).iloc[i+length-1]["filter"]]

            selected = rng.choice(len(choices), counts[source], replace=False, )

            for idx in selected:
                c, i = choices[idx]
                conv = convos.get_group(c)
                final_data.append([conv.iloc[i:i+length-1]["text"].values, c, source] + conv.iloc[i+length-1].loc["0":"27"].values.tolist())

        final_data = pd.DataFrame(final_data, columns=["text", "conversation", "source"] + [str(x) for x in range(28)])



        train, val, test = np.split(final_data.sample(frac=1, random_state=seed), 
                            [int(.8*len(final_data)), int(.9*len(final_data))])

        return train, val, test