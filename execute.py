from utility import *

if vector_size == 0:
    bert_server = None
    bert_client = None

else:
    bert_server = server.BertServer(
        server.get_args_parser().parse_args(
            [
                "-max_seq_len", "NONE",
                "-max_batch_size", "8",
                "-gpu_memory_fraction", "0.2",
                "-pooling_strategy", "NONE",
                "-pooling_layer", "-1", "-2", "-3", "-4",
                "-model_dir", bert_archive_path
            ]
        )
    )

    bert_server.start()
    bert_client = client.BertClient(output_fmt="list")

begin_time = datetime.datetime.now()
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("dataset_path")
argument_parser.add_argument("solution_path")
multiprocessing_pool = multiprocessing.Pool(psutil.cpu_count(False))

target_composite = multiprocessing_pool.map(
    func=enrich_composite,
    iterable=convert_dataset(
        load_file(argument_parser.parse_args().dataset_path, "json"),
        load_file(word_vocabulary_path, "text"),
        False
    )
)

multiprocessing_pool.close()
multiprocessing_pool.join()
target_solution = {}
SAVER = tf.train.import_meta_graph(model_graph_path)
PASSAGE_SYMBOLS = tf.get_collection("PASSAGE_SYMBOLS")[0]
PASSAGE_NUMBERS = tf.get_collection("PASSAGE_NUMBERS")[0]
PASSAGE_VECTORS = tf.get_collection("PASSAGE_VECTORS")[0]
PASSAGE_CONNECTIONS = tf.get_collection("PASSAGE_CONNECTIONS")[0]
QUESTION_SYMBOLS = tf.get_collection("QUESTION_SYMBOLS")[0]
QUESTION_NUMBERS = tf.get_collection("QUESTION_NUMBERS")[0]
QUESTION_VECTORS = tf.get_collection("QUESTION_VECTORS")[0]
QUESTION_CONNECTIONS = tf.get_collection("QUESTION_CONNECTIONS")[0]
MODEL_PREDICT = tf.get_collection("MODEL_PREDICT")[0]
MODEL_SMOOTH = tf.get_collection("MODEL_SMOOTH")[0]

with tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
) as SESSION:
    SAVER.restore(sess=SESSION, save_path=model_storage_path)
    SESSION.run(MODEL_SMOOTH)

    for record in target_composite:
        feed_record = preload_composite([record], bert_client)[0]

        feed_dict = {
            PASSAGE_SYMBOLS: feed_record["passage_symbols"],
            PASSAGE_NUMBERS: feed_record["passage_numbers"],
            PASSAGE_VECTORS: feed_record["passage_vectors"],
            PASSAGE_CONNECTIONS: feed_record["passage_connections"],
            QUESTION_SYMBOLS: feed_record["question_symbols"],
            QUESTION_NUMBERS: feed_record["question_numbers"],
            QUESTION_VECTORS: feed_record["question_vectors"],
            QUESTION_CONNECTIONS: feed_record["question_connections"]
        }

        model_predict = SESSION.run(fetches=MODEL_PREDICT, feed_dict=feed_dict)

        target_solution[feed_record["question_id"]] = spacy_nlp(
            feed_record["passage_string"]
        )[model_predict[0]:model_predict[1] + 1].text

dump_data(target_solution, argument_parser.parse_args().solution_path, "json")

print(
    "execute: cost {} seconds to generate {} for {}".format(
        int((datetime.datetime.now() - begin_time).total_seconds()),
        argument_parser.parse_args().solution_path,
        argument_parser.parse_args().dataset_path
    )
)

if bert_client is not None:
    bert_client.close()

if bert_server is not None:
    bert_server.close()
