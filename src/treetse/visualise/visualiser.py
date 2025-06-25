import pandas as pd
from plotnine import *
import math

class Visualiser:
    def __init__(self) -> None:
        self.data = None

    def load_dataset(self, results: pd.DataFrame) -> bool:
        self.data = results

    def visualise_slope(self, results: pd.DataFrame):
        # X-axis: Acc, Gen
        # Y-axis: surprisal
        filtered_df = results[
            results['alternative'].notna() & (results['alternative'].str.strip() != '')
        ]
        print("Number of filtered results: ", len(filtered_df))
        print(filtered_df.head())
        label_probs = filtered_df['label_prob']
        alt_probs = filtered_df['alternative_prob']

        filtered_df['subject_id'] = filtered_df.index

        # Melt the dataframe
        df_long = pd.melt(filtered_df,
            id_vars='subject_id',
            value_vars=['label_prob', 'alternative_prob'],
            var_name='source',
            value_name='log_prob'
        )

        # Map source to fixed x-axis labels
        df_long['x_label'] = df_long['source'].map({
            'label_prob': 'Gen',
            'alternative_prob': 'Acc'
        })

        surprisal = lambda p: -math.log2(p)
        confidence = lambda p: math.log2(p)
        df_long['surprisal'] = df_long['log_prob'].apply(confidence)
        print(df_long.head())

        p = (
            ggplot(df_long, aes(x='x_label', y='surprisal', color='x_label')) +
            scale_x_discrete(limits=['Gen', 'Acc']) +
            geom_boxplot(aes(group='x_label'),
                            width=0.2,
                            alpha=0.4, 
                            size=0.6, 
                            outlier_shape=None, 
                            show_legend=False, 
                            position=position_nudge(x=-0.2)) +
            geom_jitter(width=0.05, size=2, alpha=0.7) +
            geom_line(aes(group='subject_id'), color='gray', alpha=0.7) +
            theme_bw() +
            theme(
                axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(8, 4),
                legend_position='right'
            ) +
            labs(x='Syntactic Construction', y='Confidence', title='How confident is a model in the negated genitive?')
        )
        p.save('scripts/output/gen_vs_acc.png', width=6, height=4, dpi=300)


    



