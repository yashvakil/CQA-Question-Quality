import numpy as np
import pandas as pd
import os
import xml.etree.ElementTree as et

from google.cloud import bigquery
from config import args


def get_dataset(dirpath: str, force_update: bool = False, debug: bool = False) -> type(pd.DataFrame):
    """

    :param filepath: The filepath where the csv dataset is stored
    :param force_update: Forcefully get a new processed dataset
    :param debug: Print debug statements
    :return: Pandas dataframe containing the dataset
    """
    filepath = os.path.join(dirpath, "Posts_Processed.csv")
    if (not os.path.exists(filepath)) or force_update:
        df = build_dataset_combine(debug)

        if debug:
            print("Saving Dataset")
        df.to_csv(filepath, index=False)
        return df

    if debug:
        print("Reading Processed Dataset")

    df = pd.read_csv(filepath)
    df = typecast_dataset(df, debug)
    return df


def build_dataset_bigquery(debug: bool = False) -> type(pd.DataFrame):
    """

    :param debug: Print debug statements
    :return: pandas Dataframe containing the dataset
    """

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./CQA System-9eecfbe53496.json"

    client = bigquery.Client()
    query = """
        SELECT id,accepted_answer_id,score,view_count,body,title,tags,answer_count,comment_count,favorite_count,closed_date
        FROM `bigquery-public-data.stackoverflow.posts_questions` 
        WHERE EXTRACT(YEAR FROM creation_date)=2018
            AND EXTRACT(MONTH FROM creation_date)>=10
            AND EXTRACT(MONTH FROM creation_date)<=12
            AND last_edit_date IS NULL
            AND view_count > 0
    """

    if debug:
        print("Firing Query")

    out_df = client.query(query).result().to_dataframe()

    if debug:
        print("Dataset Obtained")

    out_df.rename(columns={
        'id': 'Id',
        'accepted_answer_id': 'AcceptedAnswerId',
        'score': 'Score',
        'view_count': 'ViewCount',
        'body': 'Body',
        'title': 'Title',
        'tags': 'Tags',
        'answer_count': 'AnswerCount',
        'comment_count': 'CommentCount',
        'favorite_count': 'FavoriteCount',
        'closed_date': 'ClosedDate'
    }, inplace=True)

    out_df = typecast_dataset(out_df, debug)
    out_df = label_dataset(out_df, debug)
    return out_df


def build_dataset_xml(debug: bool = False) -> type(pd.DataFrame):
    """

    :param debug: Print debug statements
    :return: pandas Dataframe containing the dataset
    """
    if debug:
        print("Reading XML")

    xtree = et.parse(os.path.join(args.dataset, "xml", "Posts.xml"))
    xroot = xtree.getroot()

    df_cols = ['Id', 'PostTypeId', 'AcceptedAnswerId', 'CreationDate', 'Score', 'ViewCount',
               'Body', 'Title', 'Tags', 'AnswerCount', 'CommentCount', 'FavoriteCount', 'ClosedDate']

    rows = []
    if debug:
        print("Converting XML to CSV")

    for node in xroot:
        rows.append({
            'Id': node.attrib.get('Id'),
            'PostTypeId': node.attrib.get('PostTypeId'),
            'AcceptedAnswerId': node.attrib.get('AcceptedAnswerId'),
            'CreationDate': node.attrib.get('CreationDate'),
            'Score': node.attrib.get('Score'),
            'ViewCount': node.attrib.get('ViewCount'),
            'Body': node.attrib.get('Body'),
            'Title': node.attrib.get('Title'),
            'Tags': node.attrib.get('Tags'),
            'AnswerCount': node.attrib.get('AnswerCount'),
            'CommentCount': node.attrib.get('CommentCount'),
            'FavoriteCount': node.attrib.get('FavoriteCount'),
            'ClosedDate': node.attrib.get('ClosedDate')
        })

    out_df = pd.DataFrame(rows, columns=df_cols)
    out_df = typecast_dataset(out_df, debug)

    if debug:
        print("Pruning Dataset")

    # Dataset Pruning
    out_df = out_df[out_df['PostTypeId'] == '1']

    out_df = out_df[['Id', 'AcceptedAnswerId', 'Score', 'ViewCount', 'Body', 'Title', 'Tags',
                     'AnswerCount', 'CommentCount', 'FavoriteCount', 'ClosedDate']]
    out_df = label_dataset(out_df, debug)

    return out_df


def build_dataset_combine(debug: bool = False) -> type(pd.DataFrame):
    """

    :param debug: Print debug statements
    :return: pandas Dataframe containing the dataset
    """

    """ STACK OVERFLOW QUERY
    
        WITH MyCTE AS(
            SELECT ROW_NUMBER() OVER(ORDER BY Id ASC) AS RowNum, *
            FROM POSTS
            WHERE PostTypeId = 1 
                AND CreationDate BETWEEN '10/01/2018 00:00:01' and '12/31/2018 23:59:59'
                AND LastEditDate IS NULL
                AND ViewCount > 0
                AND DeletionDate IS NULL
        )
        SELECT Id,AcceptedAnswerId,Score,ViewCount,Body,Title,Tags,AnswerCount,CommentCount,FavoriteCount,ClosedDate
        FROM MyCTE
        WHERE RowNum >160000 AND RowNum <=200000
    """

    out_df = pd.DataFrame(columns=['Id', 'AcceptedAnswerId', 'Score', 'ViewCount', 'Body', 'Title',
                                   'Tags', 'AnswerCount', 'CommentCount', 'FavoriteCount', 'ClosedDate'])

    for i in range(1, 6):
        if debug:
            print("Reading Query Result " + str(i))
        temp_df = pd.read_csv(os.path.join(args.dataset, "combine", "QueryResults (" + str(i) + ").csv"))
        out_df = pd.concat([out_df, temp_df])

    out_df = typecast_dataset(out_df, debug)
    out_df = label_dataset(out_df, debug)

    return out_df


def typecast_dataset(df: type(pd.DataFrame()), debug: bool = False) -> type(pd.DataFrame):
    """

    :param df:
    :param debug: Print debug statements
    :return: A typecasted Pandas Dataframe containing the dataset
    """
    df['AnswerCount'] = df['AnswerCount'].replace([None], ['0'])
    df['CommentCount'] = df['CommentCount'].replace([None], ['0'])
    df['FavoriteCount'] = df['FavoriteCount'].replace([None], ['0'])
    df['Score'] = df['Score'].replace([None], ['0'])
    df['ViewCount'] = df['ViewCount'].replace([None], ['0'])

    return df.astype({
        'Id': 'int64',
        'Score': 'int32',
        'ViewCount': 'int32',
        'AnswerCount': 'int32',
        'CommentCount': 'int32',
        'FavoriteCount': 'int32',
        'ClosedDate': 'datetime64[ns]'
    })


def label_dataset(df: type(pd.DataFrame()), debug: bool = False) -> type(pd.DataFrame):
    """

    :param df: The CQA dataset to typecast
    :param debug:
    :return:
    """

    def calc_piscore(row):
        return row['Score'] / row['ViewCount'] if row['ViewCount'] != 0 else 0

    if debug:
        print("Calculating PI Score")

    df['PI'] = df.apply(calc_piscore, axis=1)
    avg_pi = df['PI'].mean()

    def label(row):
        if row['PI'] >= avg_pi:
            if pd.isnull(row['AcceptedAnswerId']):
                return 'Good'
            else:
                return 'Very Good'
        else:
            if pd.isnull(row['ClosedDate']):
                return 'Bad'
            else:
                return 'Very Bad'

    if debug:
        print("Labeling Quality")

    df['Quality'] = df.apply(label, axis=1)

    return df


# get_dataset(args.dataset, True, True)
