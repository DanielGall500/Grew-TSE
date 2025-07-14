import pandas as pd
from plotnine import labs, theme, theme_bw, guides, position_nudge, aes, geom_violin, geom_boxplot, geom_line, geom_jitter, scale_x_discrete, ggplot
from pathlib import Path
import math


class Visualiser:
    def __init__(self) -> None:
        self.data = None

    def load_dataset(self, results: pd.DataFrame) -> bool:
        self.data = results

    def visualise_slope(
        self,
        path: Path,
        results: pd.DataFrame,
        target_x_label: str,
        alt_x_label: str,
        x_axis_label: str,
        y_axis_label: str,
        title: str,
    ):
        lsize = 0.65
        fill_alpha = 0.7

        # X-axis: Acc, Gen
        # Y-axis: surprisal
        filtered_df = results[
            results["alternative"].notna() & (results["alternative"].str.strip() != "")
        ]
        print("Number of filtered results: ", len(filtered_df))
        print(filtered_df.head())

        filtered_df["subject_id"] = filtered_df.index

        print(filtered_df.head())

        # Melt the dataframe
        df_long = pd.melt(
            filtered_df,
            id_vars=["subject_id", "num_tokens"],
            value_vars=["label_prob", "alternative_prob"],
            var_name="source",
            value_name="log_prob",
        )

        # Map source to fixed x-axis labels
        df_long["x_label"] = df_long["source"].map(
            {"label_prob": target_x_label, "alternative_prob": alt_x_label}
        )

        print(df_long.head())

        def surprisal(p: float) -> float:
            return -math.log2(p)

        def confidence(p: float) -> float:
            return math.log2(p)

        df_long["surprisal"] = df_long["log_prob"].apply(confidence)
        print(df_long.head())

        p = (
            ggplot(df_long, aes(x="x_label", y="surprisal", fill="x_label"))
            + scale_x_discrete(limits=[target_x_label, alt_x_label])
            + geom_jitter(
                aes(color="x_label", size="num_tokens"), width=0.01, alpha=0.7
            )
            +
            # geom_text(aes(label='label'), nudge_y=0.1) +
            geom_line(aes(group="subject_id"), color="gray", alpha=0.7, size=0.2)
            + geom_boxplot(
                df_long[df_long["x_label"] == target_x_label],
                aes(x="x_label", y="surprisal", group="x_label"),
                width=0.2,
                alpha=0.4,
                size=0.6,
                outlier_shape=None,
                show_legend=False,
                position=position_nudge(x=-0.2),
            )
            + geom_boxplot(
                df_long[df_long["x_label"] == alt_x_label],
                aes(x="x_label", y="surprisal", group="x_label"),
                width=0.2,
                alpha=0.4,
                size=0.6,
                outlier_shape=None,
                show_legend=False,
                position=position_nudge(x=0.2),
            )
            + geom_violin(
                df_long[df_long["x_label"] == target_x_label],
                aes(x="x_label", y="surprisal", group="x_label"),
                position=position_nudge(x=-0.4),
                style="left-right",
                alpha=fill_alpha,
                size=lsize,
            )
            + geom_violin(
                df_long[df_long["x_label"] == alt_x_label],
                aes(x="x_label", y="surprisal", group="x_label"),
                position=position_nudge(x=0.4),
                style="right-left",
                alpha=fill_alpha,
                size=lsize,
            )
            + guides(fill=False)
            + theme_bw()
            + theme(figure_size=(8, 4), legend_position="none")
            + labs(x=x_axis_label, y=y_axis_label, title=title)
        )
        p.save(path, width=14, height=8, dpi=300)
