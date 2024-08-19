import pandas as pd


def postsList(filename, column):
    print(f"Initiating Data Preprocessing for {column} of {filename} ")
    data = pd.read_csv(filename, low_memory=False)
    content_data = data[column].to_list()
    content_data = remove_duplicates(content_data)
    content_data = remove_null(content_data)
    content_data = remove_non_strings(content_data)
    print("Preprocessing Complete")
    return content_data


def remove_duplicates(dataList):
    print("Removing Duplicates")
    return list(set(dataList))


def remove_null(dataList):
    print("Removing Null Values")
    return [x for x in dataList if str(x) != '']


def remove_non_strings(dataList):
    print("Removing Not String Values")
    return [x for x in dataList if isinstance(x, str)]

