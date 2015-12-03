from data_prep import load_data, transform_data


if __name__ == '__main__':
    df = load_data()
    df = transform_data(df)
