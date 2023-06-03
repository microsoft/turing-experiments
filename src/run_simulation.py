"""Run simulation and save outputs."""

import sys
import time
import datetime
import pathlib

from file_IO_handler import save_json
from fill_string_template import FilledString
from openai_handler import OpenAIModelSettings, call_openai_api


def save_simulation_result_to_unique_location(
    res: dict,
    save_folder: pathlib.Path,
) -> None:
    """Saves a single simulation's result and information to a unique file location.

    Args:
        res: dict containing a simulation's result and information.
        save_folder: path of folder to save res to.

    Returns:
        None
    """
    datetimeStr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    experiment_descriptor = res["input"]["experiment_descriptor"]
    prompt_descriptor = res["input"]["prompt_descriptor"]
    params_descriptor = res["model"]["params_descriptor"]
    engine = res["model"]["engine"]
    prompt_index = res["input"]["prompt"]["index"]

    save_string = save_folder.joinpath(
        f"{datetimeStr}_experiment_{experiment_descriptor}_prompt_{prompt_descriptor}_params_{params_descriptor}_engine_{engine}_promptindex_{prompt_index}.json",
    )
    save_json(obj=res, filename=save_string)
    print("\ninput, output, and model params saved to: ", save_string)
    return None


def run_single_simulation(
    filled_string: FilledString,
    model_settings: OpenAIModelSettings,
    prompt_descriptor: str,
    experiment_descriptor: str,
    seconds_to_sleep_before_query: int = 2,
    seconds_to_sleep_after_failed_query: int = 60,
    max_attempts: int = 3,
) -> dict | None:
    """Run experiment and save results.

    Args:
        filled_string: single `Filled_String`
            corresponding to experimental conditions and
            participant details for a single simulation.
        model_settings: (OpenAI_Model_Settings) to record parameters of language model.
        prompt_descriptor: descriptor for prompt.
        experiment_descriptor: descriptor for experiment.
        seconds_to_sleep_before_query: seconds to sleep before querying language model.
        seconds_to_sleep_after_failed_query: seconds to sleep after failed query.
        max_attempts: number of tries to ping the API before returning

    Returns:
        Result object containing all results and information for the experiment
        if successful, else none
    """
    prompt_string = filled_string.filled
    prompt_index = filled_string.index

    # Try to request response from language model.
    # If response fails because rate limit was hit, try again.
    for attempt_count in range(max_attempts):
        # Sleep to not hit the rate limit.
        time.sleep(seconds_to_sleep_before_query)
        try:
            # Get response from language model.
            res = call_openai_api(prompt_string, model_settings)

            # Record prompt aspects to results dict (for sanity checking).
            res["input"] = {
                "prompt": vars(
                    filled_string
                ),  # record as dictionary for json readability
                "prompt_descriptor": prompt_descriptor,
                "experiment_descriptor": experiment_descriptor,
                # "full_input": prompt_string, #TODO: remove redundant
                # "prompt_index": prompt_index, #TODO: remove redunant
                # "path_to_prompt_fills": path_to_prompt_fills, #TODO: remove unused
            }

            # Add script version to results dict (confirm that corect helper functions are used).
            res["script-version"] = "final"

            return res

        except Exception:
            print("Exception occured:", sys.exc_info())
            print(
                f"Try again in {seconds_to_sleep_after_failed_query} seconds, attempt {attempt_count}",
            )
            time.sleep(seconds_to_sleep_after_failed_query)

    print(
        f"Experienced {attempt_count} failed attempts to query {model_settings.engine} (prompt index={prompt_index})."
    )
    return None
