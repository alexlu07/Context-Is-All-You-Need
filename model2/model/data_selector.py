import sys
import pandas as pd
import numpy as np
from fastprogress import progress_bar
from sklearn.model_selection import train_test_split

class DataSelector:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)

    def select_data(self, n, length=5, max_text=128, filter=None, balance=False, seed=12345678):
        ratio = None
        if not isinstance(n, int):
            ratio = pd.Series(n)
            assert set(ratio.index) == set(pd.unique(self.df['source']))

        print("Filtering...")

        df = self.df.groupby("conversation").filter(lambda x: x["text"].count() >= length)
        df["text"] = df["text"].str.split(n=max_text).str[:max_text].str.join(" ")
        df["filter"] = 1
        if filter: df = filter(df)

        print("Calculating counts...")

        if ratio is not None:
            source_groups = df.groupby("source")

            group_counts = source_groups.apply(lambda x: sum(x.groupby("conversation").tail(-(length-1)).groupby("conversation")["filter"].sum()))
            counts = ((group_counts / ratio).min() * ratio).astype(int)
        else:
            source_groups = [("All", df)]
            counts = {"All": n}

        print("Grabbing data...")

        rng = np.random.default_rng(seed)

        final_data = []
        for source, frame in progress_bar(source_groups):
            convos = frame.groupby("conversation")
            convos_lens = convos["filter"].count()

            choices = [(c, i) for c, l in convos_lens.items() for i in range(l+1-length) if convos.get_group(c).iloc[i+length-1]["filter"]]

            choices_df = pd.DataFrame([convos.get_group(c).iloc[i+length-1].loc["0":"10"] for c, i in choices])
            num_labels = choices_df.shape[-1]
            if not ((choices_df == 0) | (choices_df == 1)).all(axis=None): 
                choices_df = choices_df > 0
                print("WARNING: Had to convert from logits to bool")


            probs = None
            if balance:                
                probs = (choices_df * 1 / np.maximum(choices_df.sum(0)-counts[source]/num_labels, sys.float_info.epsilon)).sum(1)
                probs /= probs.sum()

            selected = rng.choice(len(choices), counts[source], replace=False, p=probs)

            for idx in selected:
                c, i = choices[idx]
                conv = convos.get_group(c)

                final_data.append([conv.iloc[i:i+length-1]["text"].values, c, source] + conv.iloc[i+length-1].loc["0":"10"].values.tolist())

        final_data = pd.DataFrame(final_data, columns=["text", "conversation", "source"] + [str(x) for x in range(11)])

        train, val, test = np.split(final_data.sample(frac=1, random_state=seed), 
                            [int(.8*len(final_data)), int(.9*len(final_data))])

        return train, val, test