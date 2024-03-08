"""Fill string templates.

Functions to fill string templates return lists of Filled_String.
"""

from string import Template
import pandas as pd


class FilledString:
    """Functions to fill string templates return lists of Filled_String."""

    def __init__(
        self, string_template: str, dict_of_fills: dict | None, index_of_fills: int = 0
    ) -> None:
        """Consistent interface for recording values when filling prompt template.

        If `dict_of_fills` is None, then use `string_template` value as filled.

        Args:
            string_template: string with `$` placeholders.
            dict_of_fills: dict with keys corresponding to `$` placeholders
                with experimental conditions and participant values for running a single simulation.
            index_of_fills: index of corresponding row of pandas dataframe.
        """
        if dict_of_fills is None:
            self.filled = string_template
            self.template = ""
            self.values = dict()
            self.index = index_of_fills
        else:
            string_template_obj = Template(string_template)
            self.filled = string_template_obj.substitute(dict_of_fills)
            self.template = string_template
            self.values = dict_of_fills
            self.index = index_of_fills

    def __str__(self) -> str:
        """Enables printing object as string.

        Returns:
            Dictionary of object attributes to string.
        """
        return str(vars(self))


def get_filled_strings_from_dataframe(
    string_template: str, dataframe_of_fills: pd.DataFrame
) -> list[FilledString]:
    """Fill string template with rows of dataframe.

    Args:
        string_template: string with `$` placeholders.
        dataframe_of_fills: pandas dataframe with columns corresponding to `$` placeholders.

    Returns:
        List of Filled_String.
    """
    filled_strings: list[FilledString] = []

    index_of_fills = 0
    for _, row in dataframe_of_fills.iterrows():
        # substitute values
        dict_of_fills = row.to_dict()
        filled_string = FilledString(
            string_template=string_template,
            dict_of_fills=dict_of_fills,
            index_of_fills=index_of_fills,
        )

        filled_strings.append(filled_string)
        index_of_fills += 1

    return filled_strings
