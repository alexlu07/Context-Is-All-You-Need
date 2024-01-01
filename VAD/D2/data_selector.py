import sys
import pandas as pd
import numpy as np
from fastprogress import progress_bar
from sklearn.model_selection import train_test_split

class DataSelector:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)

    def select_data(self, n, length=5, max_text=128, filter=None, scale_sd=False, seed=42):
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
        master_bar = progress_bar(source_groups)
        for source, frame in master_bar:
            choices_df = frame.groupby("conversation").tail(-(length-1))
            choices_df = choices_df[choices_df["filter"] == True]

            probs = None
            if scale_sd:
                vad_df = choices_df[list("VAD")]
                s = scale_sd
                a = vad_df.std(0)
                m = vad_df.mean(0)
                probs = (1/s * np.exp(-(1 - s**2) * (vad_df-m)**2 / (2 * a**2 * s**2))).sum(1)

            selected = choices_df.sample(counts[source], weights=probs, random_state=rng)
            selected["text"] = selected.apply(lambda x: df.loc[x.name-5+1: x.name, "text"].tolist(), axis=1)
            final_data.append(selected)
            

        #     convos = frame.groupby("conversation")
        #     convos_lens = convos["filter"].count()

        #     choices = [(c, i) for c, l in convos_lens.items() for i in range(l+1-length) if convos.get_group(c).iloc[i+length-1]["filter"]]

        #     choices_df = pd.DataFrame([convos.get_group(c).iloc[i+length-1].loc[list("VAD")] for c, i in choices])
        #     num_labels = choices_df.shape[-1]

        #     probs = None
        #     if scale_sd:
        #         s = scale_sd
        #         a = choices_df.std(0)
        #         m = choices_df.mean(0)
        #         probs = (1/s * np.exp(-(1 - s**2) * (choices_df-m)**2 / (2 * a**2 * s**2))).sum(1)

        #     selected = rng.choice(len(choices), counts[source], replace=False, p=probs)

        #     for idx in selected:
        #         c, i = choices[idx]
        #         conv = convos.get_group(c)

        #         final_data.append([conv.loc[i:i+length, "text"].values, c, source] + conv.iloc[i+length-1].loc[list("VAD")].values.tolist())

        final_data = pd.concat(final_data).drop("filter", axis=1)
        return final_data

        # train, val, test = np.split(final_data.sample(frac=1, random_state=seed), 
        #                     [int(.8*len(final_data)), int(.9*len(final_data))])

        # return train, val, test