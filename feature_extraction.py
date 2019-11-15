import os
import pandas as pd
import nltk
import re
import textstat

from bs4 import BeautifulSoup
from data_preprocessing import get_dataset
from config import args
from nltk import sent_tokenize
nltk.download('punkt')


def get_format_details(soup: type(BeautifulSoup), debug: bool = False):
    """

    :param soup: The Beautiful soup object for parsing
    :param debug: Debug the program
    :return: 3 parameters for format details
    """
    url_count = len(soup.findAll('a'))
    paras_count = len(soup.findAll('p'))
    list_count = len(soup.findAll('ul')) + len(soup.findAll('li'))

    return url_count, paras_count, list_count


def get_text_details(text: str, debug: bool = False):
    """

    :param text:
    :param debug: Debug the program
    :return: 3 parameters for text Characters, Words and Sentences
    """

    words = nltk.word_tokenize(text, "english")
    sentences = sent_tokenize(text, "english")
    chrs = "".join(words)

    return chrs, words, sentences


def get_code_details(codes: list, debug: bool = False):
    """

    :param soup: The Beautiful soup object for parsing
    :param debug: Debug the program
    :return: 3 parametes of code Characters, Words and Sentences
    """
    code_words = []
    code_senteces = []

    if len(codes) > 0:
        for c in codes:
            code_senteces.extend(c.text.split('\n'))
            code_words.extend(c.text.split())

    code_chrs = "".join(code_words)

    return code_chrs,code_words, code_senteces


def get_readbility_scores(text: str, indices: list = ["FLESCH", "SMOG", "KINCAID", "COLEMAN", "AUTOMATED", "DALE_CHALL", "GUNNING", "DIFFICULT_WORDS"],
                           debug: bool = False) -> list:
    """

    :param text:
    :param indices: The readability indices to calculate on the text
    :param debug: Debug the program
    :return: A readability score list
    """
    params = {
        "FLESCH": "textstat.flesch_reading_ease",
        "SMOG": "textstat.smog_index",
        "KINCAID": "textstat.flesch_kincaid_grade",
        "COLEMAN": "textstat.coleman_liau_index",
        "AUTOMATED": "textstat.automated_readability_index",
        "DALE_CHALL": "textstat.dale_chall_readability_score",
        "GUNNING": "textstat.gunning_fog",
        "DIFFICULT_WORDS": "textstat.difficult_words"
    }

    scores = []

    for indice in indices:
        scores.append(eval(params[indice.upper()] + "(text)"))

    return scores


def get_body_features(soup: type(BeautifulSoup), indices: list = ["FLESCH", "SMOG", "KINCAID", "COLEMAN", "AUTOMATED", "DALE_CHALL", "GUNNING", "DIFFICULT_WORDS"],
                       debug: bool = False) -> list:
    """

    :param soup: The soup object for parsing the html body of the of the question
    :param indices: The readability indices to calculate
    :param debug: Debug the program
    :return: A feature vector list of a particular row
    """
    feature_vec = []

    codes = soup.findAll('code')

    [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
    ques_len = len(soup.getText())

    [s.extract() for s in soup('code')]
    body_txt = soup.getText()
    body_txt_len = len(body_txt)

    url_count, paras_count, list_count = get_format_details(soup, debug)
    code_chrs, code_words, code_sentences = get_code_details(codes, debug)
    txt_chrs, txt_words, txt_sentences = get_text_details(body_txt, debug)
    scores = get_readbility_scores(body_txt, indices, debug)

    upper_case_percentage = sum(1 for c in txt_chrs if c.isupper())/len(txt_chrs) if len(txt_chrs) > 0 else -100
    lower_case_percentage = sum(1 for c in txt_chrs if c.islower()) / len(txt_chrs) if len(txt_chrs) > 0 else -100

    avg_word_len = len(txt_chrs) / len(txt_words) if len(txt_words) > 0 else -100
    avg_sentence_len = len(txt_words) / len(txt_sentences) if len(txt_sentences) > 0 else -100

    code_percentage = len(code_sentences)/len(txt_sentences) if len(txt_sentences) > 0 else -100

    feature_vec.extend([url_count, paras_count, list_count, ques_len, body_txt_len, avg_word_len, avg_sentence_len,
                        upper_case_percentage, lower_case_percentage, code_percentage])
    feature_vec.extend(scores)

    return feature_vec


def get_features_dataset(dirpath: str, force_update: bool = False, debug: bool = False) -> type(pd.DataFrame):
    """

    :param dirpath: The directory path where the dataset is present
    :param force_update: Force recalculate the entire process
    :param debug: Debug the program
    :return: A Feature set dataframe
    """

    filepath = os.path.join(dirpath, "Posts_Features.csv")
    if (not os.path.exists(filepath)) or force_update:
        dataset = get_dataset(os.path.join(dirpath), force_update, debug)
        readbility_indices = {
            "FLESCH": "float64",
            "SMOG": "float64",
            "KINCAID": "float64",
            "COLEMAN": "float64",
            "AUTOMATED": "float64",
            "DALE_CHALL": "float64",
            "GUNNING": "float64",
            "DIFFICULT_WORDS": "int64"
        }

        columns = ["PostId", "NUM_TAGS", "TITLE_LEN", "URL_COUNT", "PARAGRAPH_COUNT", "LIST_COUNT",
                   "QUESTION_LEN", "BODY_TEXT_LEN", "AVG_WORD_LEN", "AVG_SENTENCE_LEN",
                   "UPPERCASE_PERCENTAGE", "LOWERCASE_PERCENTAGE", "CODE_PERCENTAGE"]
        columns.extend(list(readbility_indices.keys()))
        columns.append("Quality")

        dataset_features = []

        if debug:
            print("Creating Feature Vectors")

        for index, row in dataset.iterrows():

            soup = BeautifulSoup(row['Body'], "html.parser")
            body_features = get_body_features(soup, list(readbility_indices.keys()), debug)

            tags = list(filter(None, re.split('\<|\>',row['Tags'])))

            num_tags = len(tags)
            title_len = len(row['Title'])

            feature_vector = [row['Id'], num_tags, title_len]
            feature_vector.extend(body_features)
            feature_vector.append(row["Quality"])

            dataset_features.append(feature_vector)

        feature_df = pd.DataFrame(dataset_features, columns=columns)
        feature_df = feature_df.astype({
            'PostId': 'int64',
            "NUM_TAGS": 'int32',
            "TITLE_LEN": 'int32',
            "URL_COUNT": 'int32',
            "PARAGRAPH_COUNT": 'int32',
            "LIST_COUNT": 'int32',
            "QUESTION_LEN": 'int64',
            "BODY_TEXT_LEN": 'int64',
            "AVG_WORD_LEN": "float64",
            "AVG_SENTENCE_LEN": "float64",
            "UPPERCASE_PERCENTAGE": "float64",
            "LOWERCASE_PERCENTAGE": "float64",
            "CODE_PERCENTAGE": "float64"
        })
        feature_df = feature_df.astype(readbility_indices)

        if debug:
            print("Saving Feature Set")

        feature_df.to_csv(filepath, index=False)
        return feature_df

    if debug:
        print("Reading Feature Dataset")

    feature_df = pd.read_csv(filepath)
    return feature_df


def get_train_test(dirpath: str, force_update: bool = False, debug: bool = False):
    """

    :param dirpath: The directory path where the dataset is present
    :param force_update: Force recalculate entire process
    :param debug: Debug the program
    :return: Two dataframes, one train and the other test
    """
    train_filepath = os.path.join(dirpath, "Posts_train.csv")
    test_filepath = os.path.join(dirpath, "Posts_test.csv")

    if (not os.path.exists(train_filepath)) or (not os.path.exists(test_filepath)) or force_update:
        feaures_df = get_features_dataset(dirpath, force_update, debug)

        if debug:
            print("Splitting Feature Dataset")

        train_df = feaures_df.sample(frac=0.8, random_state=200)  # random state is a seed value
        test_df = feaures_df.drop(train_df.index)

        if debug:
            print("Saving Split Dataset")

        train_df.to_csv(train_filepath, index=False)
        test_df.to_csv(test_filepath, index=False)
    else:
        if debug:
            print("Reading Split Dataset")

        train_df = pd.read_csv(train_filepath)
        test_df = pd.read_csv(test_filepath)

    return train_df, test_df

# get_train_test(args.dataset, True, True)