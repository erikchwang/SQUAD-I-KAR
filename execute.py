from utility import *

begin_time = datetime.datetime.now()
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("dataset_path")
argument_parser.add_argument("solution_path")
multiprocessing_pool = multiprocessing.Pool()

target_composite = multiprocessing_pool.map(
    func=enrich_composite,
    iterable=convert_dataset(
        load_file(argument_parser.parse_args().dataset_path, "json"),
        load_file(vocabulary_catalog_path, "text"),
        False
    )
)

multiprocessing_pool.close()
multiprocessing_pool.join()
target_solution = {}
SAVER = tf.train.import_meta_graph(model_graph_path)
PASSAGE_SYMBOLS = tf.get_collection("PASSAGE_SYMBOLS")[0]
PASSAGE_NUMBERS = tf.get_collection("PASSAGE_NUMBERS")[0]
PASSAGE_PASSAGE_INDICES = tf.get_collection("PASSAGE_PASSAGE_INDICES")[0]
PASSAGE_PASSAGE_VALUES = tf.get_collection("PASSAGE_PASSAGE_VALUES")[0]
PASSAGE_PASSAGE_SHAPE = tf.get_collection("PASSAGE_PASSAGE_SHAPE")[0]
QUESTION_SYMBOLS = tf.get_collection("QUESTION_SYMBOLS")[0]
QUESTION_NUMBERS = tf.get_collection("QUESTION_NUMBERS")[0]
QUESTION_PASSAGE_INDICES = tf.get_collection("QUESTION_PASSAGE_INDICES")[0]
QUESTION_PASSAGE_VALUES = tf.get_collection("QUESTION_PASSAGE_VALUES")[0]
QUESTION_PASSAGE_SHAPE = tf.get_collection("QUESTION_PASSAGE_SHAPE")[0]
MODEL_PREDICT = tf.get_collection("MODEL_PREDICT")[0]
MODEL_AVERAGE = tf.get_collection("MODEL_AVERAGE")[0]

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as SESSION:
    SAVER.restore(sess=SESSION, save_path=model_storage_path)
    SESSION.run(MODEL_AVERAGE)

    for record in target_composite:
        predict = SESSION.run(
            fetches=MODEL_PREDICT,
            feed_dict={
                PASSAGE_SYMBOLS: record["passage_symbols"],
                PASSAGE_NUMBERS: record["passage_numbers"],
                PASSAGE_PASSAGE_INDICES: record["passage_passage_indices"],
                PASSAGE_PASSAGE_VALUES: record["passage_passage_values"],
                PASSAGE_PASSAGE_SHAPE: record["passage_passage_shape"],
                QUESTION_SYMBOLS: record["question_symbols"],
                QUESTION_NUMBERS: record["question_numbers"],
                QUESTION_PASSAGE_INDICES: record["question_passage_indices"],
                QUESTION_PASSAGE_VALUES: record["question_passage_values"],
                QUESTION_PASSAGE_SHAPE: record["question_passage_shape"]
            }
        ).tolist()

        target_solution[record["question_id"]] = spacy_nlp(record["passage_source"])[predict[0]:predict[1] + 1].text

dump_data(target_solution, argument_parser.parse_args().solution_path, "json")

print(
    "execute: cost {} seconds to generate {} for {}".format(
        int((datetime.datetime.now() - begin_time).total_seconds()),
        argument_parser.parse_args().solution_path,
        argument_parser.parse_args().dataset_path
    )
)
