
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use("ggplot")

from plotnine import *
from pyprojroot import here
from sklearn.metrics import mean_absolute_error

def scatter_predicted_chronological(df:pd.DataFrame, study:str):
    min_age = min(df['chronological_age'])
    median_age = np.median(df['chronological_age'])
    max_age = max(df['chronological_age'])
    mae = mean_absolute_error(df['chronological_age'], df['brain_age'])
    site = study

    text_y = max_age + 0.5
    text_x = max_age - 1

    gg = (
        ggplot(df, aes(x="chronological_age", y="brain_age", color="brain_age_gap"))
        + geom_point(size=3, alpha=0.7)
        + scale_color_continuous(cmap_name="inferno")  # use viridis inferno palette
        + geom_smooth(
            method="lm",
            se=False,
            #color="#333333",  # dark gray line
            color= "#800080",
            alpha=0.5,
            linetype="dashed",
            size=1.5,
        )
        + labs(
            #title="Predicted Age vs Chronological Age",
            x="Chronological Age (Years)",
            y="Brain Age (Years)",
            color="Brain Age Gap",  # label for color legend
        )
        + annotate(
            "text",
            x=text_x,
            y=text_y,
            label=f"r={df['chronological_age'].corr(df['brain_age']):.3f}\nmae={mae:.3f}",
            size=14,
            fontstyle="italic",
            color="#333333",  # match line color
        )
        + theme_minimal()
        + theme(
            figure_size=(8, 8),
            panel_background=element_rect(fill="#F7F7F7"),  # light gray background
            panel_grid_major=element_line(color="#CCCCCC", size=0.5),
            panel_grid_minor=element_line(color="#CCCCCC", size=0.2),
            axis_line=element_line(color="#666666", size=1),
            axis_text=element_text(color="#666666", size=12),
            axis_title=element_text(color="#333333", size=14),
            plot_title=element_text(color="#333333", size=18, hjust=0.5),
        )
        + xlim(8, 28)
        + ylim(5, 28)
    )

    print(f"min age: {min_age}, median: {median_age}, max: {max_age}")
    print(gg)
    gg.save(f"{here()}/figures/{site}_chronological_predicted_plot.png", dpi=800)

def scatter_bag_age(df: pd.DataFrame, study: str):
    min_age = min(df['chronological_age'])
    median_age = np.median(df['chronological_age'])
    max_age = max(df['chronological_age'])  
    site = study 

    p1 = (
        ggplot(df, aes(x='chronological_age', y='brain_age_gap')) +
        geom_point() +
        geom_hline(yintercept=0, size=1, linetype='--') +
        xlab('Chronological Age (Years)') +
        ylab('Brain Age Gap (Years)') +
        stat_smooth(method='lm', color="blue") +
        theme_minimal() +
        xlim(8, 28) +
        ylim(-12, 12)
    )

    print(f"min age: {min_age}, median: {median_age}, max: {max_age}")
    print(p1)  
    p1.save(f"{here()}/figures/{site}_bag_age_plot.png", dpi=800)

def scatter_adjbag_age(df:pd.DataFrame):
    min_age = min(df['chronological_age'])
    median_age = np.median(df['chronological_age'])
    max_age = max(df['chronological_age'])   

    p1 = ggplot(aes(df['chronological_age'],df['adjusted_brain_age_gap'])) + geom_point() +\
    geom_hline(yintercept=[0], size=1, linetype='--') +\
    xlab('Chronological Age (Years)') +\
    ylab('Adjusted Brain Age Gap (Years)') +\
    stat_smooth(method='lm', color='blue') +\
    theme_minimal() +\
    xlim(8, 28) +\
    ylim(-12, 12)

    print(f"min age: {min_age}, median: {median_age}, max: {max_age}")
    print(p1) 

def scatter_adjbrainage_age(df:pd.DataFrame):
    min_age = min(df['chronological_age'])
    median_age = np.median(df['chronological_age'])
    max_age = max(df['chronological_age'])  
    mae = mean_absolute_error(df['chronological_age'], df['brain_age'])

    text_y = max_age + 0.5
    text_x = max_age - 1

    gg = (
        ggplot(df, aes(x="chronological_age", y="adjusted_brain_age", color="adjusted_brain_age_gap"))
        + geom_point(size=3, alpha=0.7)
        + scale_color_continuous(cmap=plt.cm.inferno)  # use viridis inferno palette
        + geom_smooth(
            method="lm",
            se=False,
            color="#333333", 
            alpha=0.5,
            linetype="dashed",
            size=1.5,
        )
        + labs(
            #title="Predicted Age (Bias Corrected) vs Chronological Age",
            x="Chronological Age (Years)",
            y="Adjusted Brain Age (Years)",
            color="Adjusted Brain Age Gap",  # label for color legend
        )
        + annotate(
            "text",
            x=text_x,
            y=text_y,
            label=f"r={df['chronological_age'].corr(df['brain_age']):.3f}\nmae={mae:.3f}",
            size=14,
            fontstyle="italic",
            color="#333333",  # match line color
        )
        + theme_minimal()
        + theme(
            figure_size=(8, 8),
            panel_background=element_rect(fill="#F7F7F7"),  # light gray background
            panel_grid_major=element_line(color="#CCCCCC", size=0.5),
            panel_grid_minor=element_line(color="#CCCCCC", size=0.2),
            axis_line=element_line(color="#666666", size=1),
            axis_text=element_text(color="#666666", size=12),
            axis_title=element_text(color="#333333", size=14),
            plot_title=element_text(color="#333333", size=18, hjust=0.5),
        )
    )

    print(f"min age: {min_age}, median: {median_age}, max: {max_age}")
    print(gg)
    #gg.save("../figures/frb_chronological_adjusted_predicted_plot.png", dpi=800)

def validation_error_plot(reg, eta):
    import plotnine as p9
    import pandas as pd

    # Create a pandas dataframe from the data
    df = pd.DataFrame({
        'boosting_round': range(1, len(reg.evals_result()['validation_1']['mae']) + 1),
        'mae': reg.evals_result()['validation_1']['mae']
    })

    # Find the best MAE and corresponding boosting round
    best_mae = df['mae'].min()
    std = df['mae'].std()
    best_round = df.loc[df['mae'].idxmin(), 'boosting_round']

    best_df = pd.DataFrame({'boosting_round': [best_round], 'mae': [best_mae]})

    # Create a separate dataframe for the annotation
    annot_df = pd.DataFrame({
        'boosting_round': [best_round],
        'mae': [best_mae],
        'label': [f'learning_rate={eta}\nbest_iteration={best_round}\nbest_mae={best_mae:.2f} ({std:.2f})']
    })

    # Create the plot
    g = p9.ggplot(df, p9.aes(x='boosting_round', y='mae'))
    g += p9.geom_line(color='#3498db')
    g += p9.geom_point(data=best_df, color='red', size=3)
    g += p9.geom_label(p9.aes(label='label'), data=annot_df, 
                x=float('inf'), y=float('inf'), ha='right', va='top')
    g += p9.labs(title='Validation Scores by Boosting Round', 
                x='Boosting Round', y='MAE')
    g += p9.theme_classic()
    g += p9.theme(legend_position='none')

    print(g)
    return g

def feature_importance_plot(df:pd.DataFrame):

    bar_plot = (ggplot(df, aes(x='feature_name', y='feature_importance')) 
                + geom_col(fill='#337AB7') 
                + theme_classic() 
                + coord_flip()
                + labs(y='Feature Importance', x=' ')  
                + theme(axis_text_x=element_text(angle=45, hjust=1),  
                        panel_border=element_rect(colour="black", fill=None, size=1),  
                        plot_title=element_text(ha="center"))  
                + ggtitle(" ") 
            )
    
    print(bar_plot)

    return bar_plot

    #bar_plot.save(f"{here()}/figures/{sex}_feature_importance_plot.png", dpi=800)


def age_distribution_plot(df:pd.DataFrame, study:str, color:str):
    plot = (
    ggplot(df, aes(x='chronological_age')) +
    geom_histogram(aes(y='..density..'), fill=f'{color}', bins=15, alpha=0.6, color='black') + 
    geom_density(color='black', alpha=0.5) + 
    xlab('Chronological Age') +
    ylab('Density') + 
    #ggtitle('Distribution of Age') +
    theme_minimal() +
    xlim(8, 28) +
    ylim(0, 0.17) 
    )
    print(plot)
    plot.save(f"{here()}/figures/{study}_age_distribution.png", dpi=800)

def brainage_distribution_plot(df:pd.DataFrame, study:str, color:str):
    plot = (
    ggplot(df, aes(x='brain_age')) +
    geom_histogram(aes(y='..density..'), fill=f'{color}', bins=15, alpha=0.6, color='black') + 
    geom_density(color='black', alpha=0.5) + 
    xlab('Brain Age') +
    ylab('Density') + 
    #ggtitle('Distribution of Age') +
    theme_minimal() +
    xlim(8, 28) +
    ylim(0, 0.17) 
    )
    print(plot)
    plot.save(f"{here()}/figures/{study}_brainage_distribution.png", dpi=800)

def age_distribution_plot_bhrc(df:pd.DataFrame, study:str, color:str):
    plot = (
    ggplot(df, aes(x='chronological_age')) +
    geom_histogram(aes(y='..density..'), fill=f'{color}', bins=15, alpha=0.6, color='black') + 
    geom_density(color='black', alpha=0.5) + 
    xlab('Chronological Age') +
    ylab('Density') + 
    #ggtitle('Distribution of Age') +
    theme_minimal() +
    xlim(3, 25) +
    ylim(0, 0.17) 
    )

    print(plot)
    plot.save(f"{here()}/figures/{study}_age_distribution.png", dpi=800)


def scatter_predicted_chronological_amap(df:pd.DataFrame):
    min_age = min(df['chronological_age'])
    median_age = np.median(df['chronological_age'])
    max_age = max(df['chronological_age'])
    mae = mean_absolute_error(df['chronological_age'], df['brain_age'])

    text_y = max_age + 0.5
    text_x = max_age - 1

    gg = (
        ggplot(df, aes(x="chronological_age", y="brain_age", color="brain_age_gap"))
        + geom_point(size=3, alpha=0.7)
        + scale_color_continuous(cmap_name=plt.cm.inferno)  # use viridis inferno palette
        + geom_smooth(
            method="lm",
            se=False,
            color= "#800080", #"#ff0000", #"#ff00ff",
            alpha=0.7,
            linetype="solid",
            size=1.5,
        )
        + geom_point(size=3, alpha=0.7)
        #+ geom_abline(intercept=0, slope=1, color='red', linetype='solid', size=1)  # Add red diagonal line
        + labs(
            #title="Predicted Age vs Chronological Age",
            x="Chronological Age (Years)",
            y="Brain Age (Years)",
            color="Brain Age Gap",  # label for color legend
        )
        + annotate(
            "text",
            x=text_x,
            y=text_y,
            label=f"r={df['chronological_age'].corr(df['brain_age']):.3f}\nmae={mae:.3f}",
            size=14,
            fontstyle="italic",
            color="#333333",  # match line color
        )
        + theme_void()
        + scale_fill_manual(values=["##f2f2f2", "#4682B4"])
        + theme(
            figure_size=(8, 8),
            axis_text_x=element_text(angle=0, hjust=1, margin={'t': 5}),
            axis_text_y=element_text(angle=0, hjust=1, margin={'t': 3}),
            axis_title_x=element_text(vjust=-0.5),
            axis_title_y=element_text(angle=90, margin={'l': 5, 'r': 3}),
            axis_line=element_line(color="#666666", size=1),
            axis_text=element_text(color="#666666", size=12),
            axis_title=element_text(color="#333333", size=14),
            plot_title=element_text(color="#333333", size=18, hjust=0.5),
            panel_background=element_rect(fill="#FFFFFF", color="#FFFFFF"),
        )
    )

    print(f"min age: {min_age}, median: {median_age}, max: {max_age}")
    print(gg)
    #gg.save(f"../figures/frb_predictions.png", dpi=800)



if __name__ == '__main__':

    scatter_predicted_chronological() 
    scatter_bag_age() 
    scatter_adjbag_age() 
    scatter_adjbrainage_age()
    validation_error_plot()
    feature_importance_plot()
    age_distribution_plot()

