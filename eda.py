from data_prep import load_data, transform


if __name__ == '__main__':
    df = load_data()
    df = transform(df)
