"""
Create embeddings for each document in a database, using the 'texts' field
of each document. The embeddings represent the 'source' of the texts. The
calculated embeddings are saved to the same database.
"""
import json
import logging
from os import path
from person2vec import data_handler
from person2vec.generators import training_data_generator
from person2vec.utils import snippet_creator
from person2vec.train_embeddings import train

logging.basicConfig(
    format='%(asctime)-15s [%(levelname)s:%(module)s] %(message)s',
    level=logging.INFO)
logger = logging.getLogger()

HERE = path.abspath(path.dirname(__file__))
PROJECT_DIR = path.dirname(HERE)
DATA_DIR = path.join(PROJECT_DIR, 'person2vec', 'data')


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        'Create embeddings for any target that you can provide text(s). '
        'Target-texts pairs should be on a Mongo DB.\n'
        'Provide the connection parameters. '
        'If none is given then an attempt to read from '
        '"data/db_settings.json" is made.')
    parser.add_argument('-u', '--user', help='Database user name.')
    parser.add_argument('-p', '--pwd', help='Database password.')
    parser.add_argument(
        '-t',
        '--host',
        default='localhost',
        help='IP or URL of the MongoDB server. Default "localhost"')
    parser.add_argument(
        '-r',
        '--port',
        default='27017',
        help='The port of the MongoDB server. Default "27017"')
    parser.add_argument(
        '-d', '--db_name', help='The name of the target database')
    args = parser.parse_args()
    if args.db_name is None:
        db_settings_path = path.join(DATA_DIR, 'db_settings.json')
        with open(db_settings_path) as fp:
            db = json.load(fp)
    else:
        db = vars(args)

    handler = data_handler.DataHandler(**db)
    snippet_creator.snippetize_db(handler)
    data_gen = training_data_generator.EmbeddingDataGenerator(handler)
    model, data_gen = train.train_model(epochs=40, data_gen=data_gen)
    handler.save_embeddings_to_db(model, data_gen)


if __name__ == "__main__":
    main()
