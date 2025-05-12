import glob
import json
import os

import pandas as pd

from datetime import datetime

path = os.environ.get('PROJECT_PATH', '/opt/airflow/plugins')


def predict():
    import dill

    models_file = glob.glob(f'{path}/data/models/*')
    model_file_path = max(models_file, key=os.path.getctime)

    with open(f'{model_file_path}', 'rb') as file:
        model = dill.load(file)

    test_file_name = os.listdir(f'{path}/data/test')
    df = pd.DataFrame()

    for file_name in test_file_name:
        with open(f'{path}/data/test/{file_name}', 'r') as file_test:
            test = json.load(file_test)
        df = pd.concat([df, pd.DataFrame.from_dict([test])])

    df['predict'] = model.predict(df)

    df.to_csv(f'{path}/data/predictions/predictions{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
