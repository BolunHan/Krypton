import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt

__all__ = ['df_to_png', 'df_to_table']


def df_to_png(df: pd.DataFrame, formatters=None, save_path=None, font=None) -> io.BytesIO:
    """
    Plot pandas DataFrame into a png picture
    :param df: the given pandas DataFrame
    :param formatters: dict formatter function of data, function will be map to each column with correspond name
    :param save_path: .png file save path
    :param font: the font for all characters
    :return: a file io object
    """
    ax = plt.subplot(frame_on=False)  # no visible frame
    # noinspection PyUnresolvedReferences
    ax.xaxis.set_visible(False)  # hide the x axis
    # noinspection PyUnresolvedReferences
    ax.yaxis.set_visible(False)  # hide the y axis
    if font is not None:
        plt.rcParams["font.family"] = font

    plt.rcParams['axes.unicode_minus'] = False
    formatted_df = df.copy()

    if formatters:
        for column in formatters:
            # noinspection PyBroadException
            try:
                formatted_df[column] = formatted_df[column].map(formatters[column])
            except Exception as _:
                pass
    t = pd.plotting.table(ax, formatted_df, loc='center')  # where df is your data frame
    t.auto_set_font_size(False)
    file_object = io.BytesIO()
    plt.savefig(file_object, format='png', bbox_inches='tight')

    if save_path:
        file_object.seek(0)
        with open(save_path, 'wb') as out:
            out.write(file_object.read())

    file_object.seek(0)
    plt.clf()
    plt.cla()
    plt.close()
    return file_object


def df_to_table(
        df: pd.DataFrame,
        formatters=None,
        save_path=None,
        **kwargs
) -> io.BytesIO:
    """
    Plot pandas DataFrame into a png picture with plotly
    :param df: the given pandas DataFrame
    :param formatters: dict formatter function of data, function will be map to each column with correspond name
    :param save_path: .png file save path
    :return: a file io object
    """
    default_theme = dict(
        line_color='darkslategray',
        header_fill_color='grey',
        row_even_color='lightgrey',
        row_odd_color='white',
        header_font=dict(color='white', size=12),
        row_font=dict(color='black', size=11),
        header_align=['left', 'center'],
        row_align=['left']
    )
    default_theme.update(kwargs.pop('theme', {}))

    formatted_df = df.copy()
    line_color = default_theme['line_color']
    header_fill_color = default_theme['header_fill_color']
    row_even_color = default_theme['row_even_color']
    row_odd_color = default_theme['row_odd_color']
    header_font = default_theme['header_font']
    row_font = default_theme['row_font']
    header_align = default_theme['header_align']
    row_align = default_theme['row_align']
    row_fill_color = [row_odd_color, row_even_color] * (len(df) // 2) + [row_odd_color] * np.mod(len(df), 2)

    if formatters:
        for column in formatters:
            # noinspection PyBroadException
            try:
                formatted_df[column] = formatted_df[column].map(formatters[column])
            except Exception as _:
                pass

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[formatted_df.index.name if formatted_df.index.name else ''] + list(formatted_df.columns),
                    line_color=line_color,
                    fill_color=header_fill_color,
                    font=header_font,
                    align=header_align
                ),
                cells=dict(
                    values=[[f'<b>{idx}</b>' for idx in formatted_df.index.to_list()]] + [formatted_df[col] for col in formatted_df.columns],
                    line_color=line_color,
                    fill_color=[row_fill_color * (len(formatted_df.columns) + 1)],
                    align=row_align,
                    font=row_font
                )
            )
        ]
    )

    img_bytes = fig.to_image(format="png", engine='kaleido', **kwargs)

    if save_path:
        with open(save_path, 'wb') as out:
            out.write(img_bytes)

    return io.BytesIO(img_bytes)
