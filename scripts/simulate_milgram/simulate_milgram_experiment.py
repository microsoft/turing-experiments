"""Simulate Milgram Shock Experiment.
Run from root directory with command python scripts/simulate_milgram/simulate_milgram_experiment.py
"""

import pandas as pd
import pathlib
import sys
import math
import re
import datetime
from tqdm import tqdm
import fire

# Now davinci-text-002 is deprecated so calling the LLM API will not work
# Instead, see how the overall logic of the simulated experimented works with mocked LLM responses.
# Set GLOBAL_VAR_MOCKED = False to call the LLM API
GLOBAL_VAR_MOCKED = True

# Add src module to path before import.
sys.path.insert(1, str(pathlib.Path("./src")))
from file_IO_handler import get_plaintext_file_contents, save_json  # noqa: E402
from openai_handler import verify_openai_access, OpenAIModelSettings  # noqa: E402
from fill_string_template import FilledString  # noqa: E402
from run_simulation import run_single_simulation  # noqa: E402

# Set path settings.
PATH_TO_SIMULATION_RESULTS: pathlib.Path = pathlib.Path(
    "./data/simulation_results/milgram/milgram_original"
)
PATH_TO_PROMPTS: pathlib.Path = pathlib.Path(
    "./data/prompt-templates/milgram/milgram_resources"
)
PATH_TO_PARTICIPANT_LIST: pathlib.Path = pathlib.Path(
    "./data/prompt-fills/milgram/df_names.csv"
)


def save_milgram_result_experiment_overview_to_unique_location(
    res,
    subject: str,
):
    """Save final experiment overview (for Milgram Experiment) to a unique location.

    Args:
        res: language model response.
        subject: name of simulated subject.

    Returns:
        None.
    """
    subject_nopunct = subject.replace(". ", "")
    datetimeStr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_string = PATH_TO_SIMULATION_RESULTS.joinpath(
        f"experiment-overview_{datetimeStr}_subject_{subject_nopunct}.json"
    )
    save_json(obj=res, filename=save_string)
    print("saved prompt to: ", save_string)
    return None


def save_milgram_LM_response_to_unique_location(
    res: dict,
    file_name_prefix: str,
) -> None:
    """Save language model response (for Milgram Experiment) to a unique location.

    Args:
        res: language model response.
        file_name_prefix: prefix of file name to save response to.

    Returns:
        None.
    """
    datetimeStr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    subject_nopunct = res["input"]["prompt"]["subject"].replace(". ", "")
    level = res["input"]["prompt"]["level"].replace(". ", "")
    num_disobedience = res["input"]["prompt"]["num_disobedience"].replace(". ", "")

    save_string = PATH_TO_SIMULATION_RESULTS.joinpath(
        f"{file_name_prefix}_{datetimeStr}_subject_{subject_nopunct}_level_{level}_num_{num_disobedience}.json"
    )

    save_json(obj=res, filename=save_string)
    print(f"\t\t {file_name_prefix} saved to: {save_string}")
    return None


def get_introduction_text():
    """Get introduction text.

    Returns:
        string with `$` placeholders.
    """
    return get_plaintext_file_contents(
        PATH_TO_PROMPTS.joinpath("DO_introduction.txt")
    )


def verbal_designation(row) -> str:
    """Translate numerical shock level to level's verbal designation.

    In the original Milgram Experiment, shock levels were grouped in sets of 4.

    Args:
        row: panda's dataframe row with column "shock level".

    Returns:
        string verbal designation.
    """
    shock_level_switch_group = int((row["shock level"] - 1) / 4)
    if shock_level_switch_group == 0:
        return "slight shock"
    if shock_level_switch_group == 1:
        return "moderate shock"
    if shock_level_switch_group == 2:
        return "strong shock"
    if shock_level_switch_group == 3:
        return "very strong shock"
    if shock_level_switch_group == 4:
        return "intense shock"
    if shock_level_switch_group == 5:
        return "extreme intensity shock"
    if shock_level_switch_group == 6:
        return "danger severe shock"
    else:
        return "XXX"


def get_level_information() -> pd.DataFrame:
    """Load dataframe of level information and add derivative columns.

    Returns:
        pandas dataframe.
    """
    df_level_information = pd.read_csv(
        PATH_TO_PROMPTS.joinpath("DO_level_information_remote-learner.txt"),
        delimiter="\t",
        index_col=0,
    )

    df_level_information["voltage level"] = df_level_information.apply(
        lambda x: x["shock level"] * 15, axis=1
    )

    df_level_information["verbal designation"] = df_level_information.apply(
        verbal_designation, axis=1
    )

    return df_level_information


def get_experimenter_reaction_to_subject_wrong_action() -> str:
    """Get experimenter reaction to wrong action on behalf of the subject.

    Returns:
        string with `$` placeholders.
    """
    return "The experimenter tells $subject to follow the correct experimental procedure. $subject should read the question and wait for the learner to select an answer. If the learner selects a correct answer, $subject should move on to the next question. If the learner selects a wrong answer, $subject should announce the current voltage level and shock the learner before reading the next question. $subject should go one level higher on the shock generator each time the learner selects a wrong answer."


def get_experimenter_reaction_to_subject_question_post_victim_no_response() -> str:
    """Get experimenter reaction to subject question post when the victim stops responding.

    Returns:
        string with `$` placeholders.
    """
    return "The experimenter instructs $subject to treat the absence of a response as a wrong answer."


def get_experimenter_prods() -> pd.DataFrame:
    """Get dataframe of experimenter prods that the experimenter says to the subject when subject hesitates or disobeys.

    Returns:
        pandas dataframe with experimenter prods.
    """
    return pd.read_csv(
        PATH_TO_PROMPTS.joinpath("DO_experimenter_prod.txt"),
        delimiter="\t",
        index_col=0,
    )


def prepare_level_prompt(dict_of_fills: dict) -> str:
    """Prepare level prompt.

    If at the start of the level (no disobediences), add instructions.
    End level prompt with `$subject`.

    Args:
        dict_of_fills: dict with columns "level" and "num_disobedience"

    Returns:
        string with `$` placeholders.
    """
    # Unpack dict of fills.
    level = dict_of_fills["level"]
    num_disobedience = dict_of_fills["num_disobedience"]

    level_prompt = ""

    if num_disobedience == 0:
        # If no instances of disobedience have been observed,
        # Then subject is at the start of the level and level information should be relayed.
        df_level_information = get_level_information()
        verbal_designation = df_level_information.loc[level]["verbal designation"]
        voltage_level = df_level_information.loc[level]["voltage level"]
        learner_response = df_level_information.loc[level]["learner response"]
        print("\t\t learner_response:", learner_response)

        level_prompt += f"\n\nThe current punishment voltage level is {verbal_designation}, {voltage_level} volts. The experimenter is sitting next to $subject. {learner_response}"

    level_prompt += "\n\n$subject"
    return level_prompt


def query_LM_to_generate_subject_response_text(
    filled_string: FilledString,
    mockedValue: str=" moves onto the next question. He questions all his life decisions.",
):
    """Query language model to generate subject response.

    Args:
        filled_string: filled template and variables to prompt the language model with.

    Returns:
        String with LLM generation.

    Raises:
        Exception: "`run_single_simulation` failed!" if calling API failed more than `max_attempt` times.
    """
    # Set model settings.
    model_settings_generate_free_form_text = OpenAIModelSettings(
        engine="text-davinci-002",
        max_tokens=40,
        temperature=1,
        n=1,
        logprobs=1,
        echo=False,
        params_descriptor="long-ans-logprobs",
    )

    # If mocking, do not call the LM
    # instead print out filled_string and return mockedValue
    if (GLOBAL_VAR_MOCKED):
        print("#############################################################################")
        print("### Got Prompt", filled_string.filled, "###")
        print("#############################################################################")
        print("*****************************************************************************")
        print("*** DID NOT CALL LLM to generate teacher response*** - MOCKED with", mockedValue, "***")
        print("*****************************************************************************")
        return mockedValue

    # Query LM.
    print("\t\t ***CALLED LM to generate teacher response***")
    res = run_single_simulation(
        filled_string=filled_string,
        model_settings=model_settings_generate_free_form_text,
        prompt_descriptor="generation",
        experiment_descriptor="do",
        seconds_to_sleep_before_query=1,
        seconds_to_sleep_after_failed_query=5,
        max_attempts=3,
    )

    if res is None:
        raise Exception("`run_single_simulation` failed!")

    save_milgram_LM_response_to_unique_location(
        res=res, file_name_prefix="subject-response"
    )

    res_text = res["output"]["choices"][0]["text"]

    print("\t\t %%%%%%%%%%%%")
    print(res_text)
    print("\t\t %%%%%%%%%%%%")

    return res_text


def query_LM_to_judge_did_not_shock(
    subject_response_text: str, dict_of_fills: dict,
    mockedValue: bool=True,
) -> bool:
    """Use language model to judge whether the subject did not shock the victim.

    Args:
        subject_response_text: value to add to `dict_of_fills`
        dict_of_fills: dict with keys corresponding to `$` placeholders

    Returns:
        prob > 0.5

    Raises:
        Exception: "`run_single_simulation` failed!" if calling API failed more than `max_attempt` times.
    """
    # Set model settings.
    model_settings_2_choice_judgment = OpenAIModelSettings(
        engine="text-davinci-002",
        max_tokens=0,
        temperature=0,
        n=1,
        logprobs=1,
        echo=True,
        params_descriptor="no-text-complete",
    )

    # Fill template string.
    dict_of_fills["action"] = subject_response_text
    prompt_string = get_plaintext_file_contents(
        PATH_TO_PROMPTS.joinpath("DO_judge_did_not_shock.txt")
    )
    filled_string = FilledString(
        string_template=prompt_string, dict_of_fills=dict_of_fills
    )

    # If mocking, do not call the LM
    # instead print out filled_string and return mockedValue
    if (GLOBAL_VAR_MOCKED):
        print("#############################################################################")
        print("### Got Prompt", filled_string.filled, "###")
        print("#############################################################################")
        print("*****************************************************************************")
        print("*** DID NOT CALL LLM *** - MOCKED judge did not shock with ", mockedValue, "***")
        print("*****************************************************************************")
        return mockedValue
    
    # Query LM.
    print("\t\t ***CALLED LM to judge did not shock***")
    res = run_single_simulation(
        filled_string=filled_string,
        model_settings=model_settings_2_choice_judgment,
        prompt_descriptor="DO_judge_did_not_shock",
        experiment_descriptor="DO_judge",
        seconds_to_sleep_before_query=1,
        seconds_to_sleep_after_failed_query=5,
        max_attempts=3,
    )

    if res is None:
        raise Exception("`run_single_simulation` failed!")

    # Get log probs of 'not shock'.
    token_logprobs = res["output"]["choices"][0]["logprobs"]["token_logprobs"][-2:]
    prob = math.exp(sum(token_logprobs))
    res["probability_did_not_shock"] = prob

    save_milgram_LM_response_to_unique_location(
        res=res, file_name_prefix="judge-response-not-shock"
    )

    print("\t\t judge-response not shock probability:", prob)
    print("\t\t ############")
    print(res["output"]["choices"][0]["text"])
    print("\t\t ############")

    return prob > 0.5


def query_LM_to_judge_did_not_stop(
    subject_response_text: str, dict_of_fills: dict,
    mockedValue: bool=True
) -> bool:
    """Use language model to judge whether the subject did not stop the experiment.

    Args:
        subject_response_text: value to add to `dict_of_fills`
        dict_of_fills: dict with keys corresponding to `$` placeholders

    Returns:
        prob > 0.5

    Raises:
        Exception: "`run_single_simulation` failed!" if calling API failed more than `max_attempt` times.
    """
    # Set model settings.
    model_settings_2_choice_judgment = OpenAIModelSettings(
        engine="text-davinci-002",
        max_tokens=0,
        temperature=0,
        n=1,
        logprobs=1,
        echo=True,
        params_descriptor="no-text-complete",
    )

    # Fill template string.
    dict_of_fills["action"] = subject_response_text
    prompt_string = get_plaintext_file_contents(
        PATH_TO_PROMPTS.joinpath("DO_judge_did_not_stop.txt")
    )
    filled_string = FilledString(
        string_template=prompt_string, dict_of_fills=dict_of_fills
    )

    # If mocking, do not call the LM
    # instead print out filled_string and return mockedValue
    if (GLOBAL_VAR_MOCKED):
        print("#############################################################################")
        print("### Got Prompt", filled_string.filled, "###")
        print("#############################################################################")
        print("*****************************************************************************")
        print("*** DID NOT CALL LLM *** MOCKED judge did not stop with ", mockedValue, "***")
        print("*****************************************************************************")
        return mockedValue

    # Query LM.
    print("\t\t ***CALLED LM to judge did not stop***")
    res = run_single_simulation(
        filled_string=filled_string,
        model_settings=model_settings_2_choice_judgment,
        prompt_descriptor="DO_judge_did_not_stop",
        experiment_descriptor="DO_judge",
        seconds_to_sleep_before_query=1,
        seconds_to_sleep_after_failed_query=5,
        max_attempts=3,
    )

    if res is None:
        raise Exception("`run_single_simulation` failed!")

    # Get log probs of 'not stop'
    token_logprobs = res["output"]["choices"][0]["logprobs"]["token_logprobs"][-2:]
    prob = math.exp(sum(token_logprobs))
    res["probability_did_not_stop"] = prob

    save_milgram_LM_response_to_unique_location(
        res=res, file_name_prefix="judge-response-not-stop"
    )

    print("\t\t judge-response not stop probability:", prob)
    print("\t\t ############")
    print(res["output"]["choices"][0]["text"])
    print("\t\t ############")

    return prob > 0.5


def get_experimenter_feedback(subject_response_text: str, dict_of_fills: dict) -> dict:
    """Return the experimenter's response to the subject's response.

    Args:
        subject_response_text: value to add to `dict_of_fills`
        dict_of_fills: dict with keys corresponding to `$` placeholders

    Returns:
        dictionary with "feedback", "finish_level", "num_disobedience", "finish_experiment"

    Cases:
    If subject did stop:
        Feedback is subject's response (experiment is finished)
    If subject did shock & if learner's action was correct:
        Feedback is subject's response + experimenter repeats instructions (level is finished)
    If subject did shock & if learner's action was wrong
        Feedback is subject's response (level is finished)
    If subject did not shock & if learner action was correct
        Feedback is subject's response (level is finished)
    If subject did not shock & learner action was wrong
        Feedback is subject's response + experimenter uses prod (level repeats)
    """
    # Unpack level information.
    level = dict_of_fills["level"]
    df_level_information = get_level_information()
    learner_response = df_level_information.loc[level]["learner response"]

    # Grab only use the first sentence of subject's response.
    RE_GRAB_FIRST_SENTENCE = '(.*?(?<!Mr|Ms)[\.\!\?\;]"?)'
    match = re.search(RE_GRAB_FIRST_SENTENCE, subject_response_text)
    first_sentence = match.group(1)
    # Edge case: if character is a semicolon, change to a period.
    first_sentence = first_sentence.replace(";", ".")

    # Ask LM to simulate experimenter deciding whether the action is following orders.
    did_not_shock = query_LM_to_judge_did_not_shock(
        subject_response_text=first_sentence, dict_of_fills=dict_of_fills
    )
    did_not_stop = query_LM_to_judge_did_not_stop(
        subject_response_text=first_sentence, dict_of_fills=dict_of_fills
    )

    # Handle cases.
    case_info = {
        "feedback": first_sentence,
        "finish_experiment": False,
        "finish_level": True,
        "num_disobedience": dict_of_fills["num_disobedience"],
    }

    # If subject did stop, end the experiment.
    if not did_not_stop:
        case_info["finish_experiment"] = True
        case_info["num_disobedience"] += 1
        return case_info

    # If subject did shock, respond depending on
    # whether the learner's action is correct or incorrect.
    if not did_not_shock:
        if learner_response == "The learner selects a correct answer.":
            # If learner action was correct, remind subject of instructions.
            # Update feedback to be subject's response + experimenter repeats instructions (level is finished).
            case_info["feedback"] += (
                "\n\n" + get_experimenter_reaction_to_subject_wrong_action()
            )
        # Else if learner's action was incorrect, then subject followed shock instructions,
        # so feedback is just the subject's response (level is finished).
        return case_info

    # If subject did not shock, respond depending on
    # whether the learner's action is correct or incorrect.
    if learner_response == "The learner selects a correct answer.":
        # If learner action was correct, then subject followed shock instructions,
        # so feedback is just the subject's response (level is finished).
        return case_info

    # Otherwise subject is disobeying shock instructions.
    # Experimenter uses an action to nudge subject into shocking (level continues).
    case_info["finish_level"] = False
    case_info["num_disobedience"] += 1

    # Handle special case (victim does not respond if level is 27).
    # If level 27 and 1 disobedience has been observed,
    # give instructions on how to handle no response.
    if level == 27 and case_info["num_disobedience"] == 1:
        case_info["feedback"] += (
            "\n\n"
            + get_experimenter_reaction_to_subject_question_post_victim_no_response()
        )
        return case_info

    # Experimenter uses prods sequentially with increasing disobedience.
    if case_info["num_disobedience"] <= 4:
        prod_name = f"prod {case_info['num_disobedience']}"
        prod_text = get_experimenter_prods().loc[prod_name]["text"]
        case_info["feedback"] += f'\n\nThe experimenter tells $subject, "{prod_text}"'
        return case_info

    # If subject is disobedient 5 times, end the experiment.
    case_info["finish_level"] = True
    case_info["finish_experiment"] = True
    case_info["feedback"] += "\n\nThe experimenter ends the experiment."
    return case_info


def run_single_level(previous_prompt_string: str, subject: str, level: int):
    """Run a single level of the Milgram experiment until it meets an end condition.

    Args:
        previous_prompt_string: string with `$` placeholders. To initialize current prompt string.
        subject: subject's name.
        level: current level.

    Returns:
        dictionary with
            new "prompt_string",
            completed "level",
            "is_end_experiment" (num_disobedience >= 5) or (finish_level and level == 36)
            "num_disobedience" observed in the level
    """
    # Prepare dict of fills.
    dict_of_fills = {
        "subject": subject,
        "pronoun": "he" if subject.startswith("Mr.") else "she",
        "poss": "his" if subject.startswith("Mr.") else "her",
        "poss_obj": "him" if subject.startswith("Mr.") else "her",  # added this
        "victim": "Mr. Wallace",
        "level": level,
        "num_disobedience": 0,
    }

    # Load intro text from file.
    if level == 1:
        prompt_string = get_introduction_text()
    else:
        prompt_string = previous_prompt_string

    # In each level, the subject responds and experimenter prods
    # repeatedly until the level ends.
    while True:
        # Add learner response for this level to the prompt.
        prompt_string += prepare_level_prompt(dict_of_fills)
        print("\t\t ------------------------------")

        # Get a subject generation from prompt.
        filled_string = FilledString(
            string_template=prompt_string, dict_of_fills=dict_of_fills
        )
        subject_response_text = query_LM_to_generate_subject_response_text(
            filled_string=filled_string
        )

        # Get experimenter feedback on subject generation.
        experimenter_feedback = get_experimenter_feedback(
            subject_response_text, dict_of_fills
        )

        # Use experimenter feedback to update prompt and fills
        # for next attempt at same level.
        prompt_string += experimenter_feedback["feedback"]
        dict_of_fills["num_disobedience"] = experimenter_feedback["num_disobedience"]

        # If level has ended...
        if experimenter_feedback["finish_level"]:
            # Calculate if experiment has ended.
            is_end_experiment = (
                experimenter_feedback["finish_experiment"]
                or (experimenter_feedback["num_disobedience"] >= 5)
                or (experimenter_feedback["finish_level"] and level == 36)
            )
            return {
                "prompt_string": prompt_string,
                "level": level,
                "is_end_experiment": is_end_experiment,
                "num_disobedience": experimenter_feedback["num_disobedience"],
            }


def run_single_experiment(
    subject: str,
    race: str,
    gender: str,
    surname: str,
) -> None:
    """Run the Milgram Experiment to completion for a single subject.

    Args:
        subject: name of subject.
        race: race of subject.
        gender: honorific of subject.
        surname: surname of subject.

    Returns:
        None.
    """
    # Lists to track progression.
    prompts = []
    levels = []
    num_disobediences = []

    # State variables for current level.
    prompt_string = ""
    level = 1
    is_end_experiment = False

    while level <= 36 and not is_end_experiment:
        print("LEVEL:", level)
        level_results = run_single_level(
            previous_prompt_string=prompt_string,
            subject=subject,
            level=level,
        )

        # Update progression.
        prompts.append(level_results["prompt_string"])
        levels.append(level_results["level"])
        num_disobediences.append(level_results["num_disobedience"])

        # Update state.
        prompt_string = level_results["prompt_string"]
        is_end_experiment = level_results["is_end_experiment"]
        if not is_end_experiment:
            level += 1

    # After experiment ends, save experiment overview.
    save_obj = {
        "subject": subject,
        "race": race,
        "gender": gender,
        "surname": surname,
        "full_prompt": prompt_string,
        "final_level": level,
        "num_disobediences": num_disobediences,
    }

    # If mocking, do not save the experiment overview, just print it
    if (GLOBAL_VAR_MOCKED):
        print("Experiment Overview:", save_obj)
        return None

    save_milgram_result_experiment_overview_to_unique_location(
        res=save_obj, subject=subject
    )

    return None


def run_full_experiment(title="Mr.", race="White") -> None:
    """Run Milgram Experiment for all participants in dataframe.

    Run script on portion of participant list (filter by title and race) to run multiple experiments in parallel.

    Args:
        title: title to run experiments with.
        race: race to filter participant list by.
    """
    # Set Language Model Settings.
    verify_openai_access(
        path_to_organization=pathlib.Path("./openai_organization.txt"),
        path_to_api_key=pathlib.Path("./openai_api_key.txt"),
    )

    full_set_df_names = pd.read_csv(PATH_TO_PARTICIPANT_LIST)
    # Run same script in parallel to save time.
    # Each script runs different set of races.
    df_names = full_set_df_names[full_set_df_names["Race"] == race]
    df_names = df_names.reset_index()

    for i, row in tqdm(df_names.iterrows()):
        subject = f'{title} {row["Surname"]}'
        print("Start", i, subject)
        run_single_experiment(
            subject=subject, race=row["Race"], gender=title, surname=row["Surname"]
        )
        break # REMOVE: for testing purposes, run 1 experiment only


if __name__ == "__main__":
    fire.Fire(run_full_experiment)
