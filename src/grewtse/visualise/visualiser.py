import pandas as pd
from plotnine import (
    labs,
    theme,
    theme_bw,
    guides,
    position_nudge,
    aes,
    geom_violin,
    geom_line,
    geom_jitter,
    scale_x_discrete,
    ggplot,
)
from pathlib import Path
import math


class GrewTSEVisualiser:
    """
    A basic visualisation class that creates a violin plot based on a syntactic evaluation.
    """

    def __init__(self) -> None:
        self.data = None

    def visualise_syntactic_performance(
        self,
        results: pd.DataFrame,
        title: str,
        target_x_label: str,
        alt_x_label: str,
        x_axis_label: str,
        y_axis_label: str,
        filename: str,
    ) -> None:
        """
        Visualise a syntactic performance evaluation result.

        :param results: pass the results DataFrame created by the GrewTSEEvaluator.
        :param title: Give the diagram a main title.
        :param target_x_label: Give the original target word and hence first word in the minimal pair a label e.g. 'Accusative'.
        :param alt_x_label: Give the second element in the minimal pair a label e.g. 'Dative'.
        :param x_axis_label: Give the X Axis a title.
        :param y_axis_label: Give the Y Axis a title.
        :param filename: A filename to save the visualisation.
        :return:
        """

        visualise_slope(
            filename,
            results,
            target_x_label,
            alt_x_label,
            x_axis_label,
            y_axis_label,
            title,
        )


def visualise_slope(
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
        results["form_ungrammatical"].notna()
        & (results["form_ungrammatical"].str.strip() != "")
    ]

    filtered_df["subject_id"] = filtered_df.index

    # Melt the dataframe
    df_long = pd.melt(
        filtered_df,
        id_vars=["subject_id"],
        value_vars=["p_grammatical", "p_ungrammatical"],
        var_name="source",
        value_name="log_prob",
    )

    # Map source to fixed x-axis labels
    df_long["x_label"] = df_long["source"].map(
        {"p_grammatical": target_x_label, "p_ungrammatical": alt_x_label}
    )

    def surprisal(p: float) -> float:
        return -math.log2(p)

    def confidence(p: float) -> float:
        return math.log2(p)

    df_long["surprisal"] = df_long["log_prob"].apply(surprisal)

    p = (
        ggplot(df_long, aes(x="x_label", y="surprisal", fill="x_label"))
        + scale_x_discrete(limits=[target_x_label, alt_x_label])
        + geom_jitter(aes(color="x_label"), width=0.01, alpha=0.7)
        +
        # geom_text(aes(label='label'), nudge_y=0.1) +
        geom_line(aes(group="subject_id"), color="gray", alpha=0.7, size=0.2)
        + geom_violin(
            df_long[df_long["x_label"] == target_x_label],
            aes(x="x_label", y="surprisal", group="x_label"),
            position=position_nudge(x=-0.2),
            style="left-right",
            alpha=fill_alpha,
            size=lsize,
        )
        + geom_violin(
            df_long[df_long["x_label"] == alt_x_label],
            aes(x="x_label", y="surprisal", group="x_label"),
            position=position_nudge(x=0.2),
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
