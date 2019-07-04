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
                "-model_dir", bert_archive_path,
                "-num_worker", "{}".format(gpu_count)
            ]
        )
    )

    bert_server.start()
    bert_client = client.BertClient(output_fmt="list")

train_composite = load_file(train_composite_path, "pickle")
develop_composite = load_file(develop_composite_path, "pickle")
SAVER = tf.train.import_meta_graph(model_graph_path)
PASSAGE_SYMBOLS_BATCH = tf.get_collection("PASSAGE_SYMBOLS_BATCH")
PASSAGE_NUMBERS_BATCH = tf.get_collection("PASSAGE_NUMBERS_BATCH")
PASSAGE_VECTORS_BATCH = tf.get_collection("PASSAGE_VECTORS_BATCH")
PASSAGE_CONNECTIONS_BATCH = tf.get_collection("PASSAGE_CONNECTIONS_BATCH")
QUESTION_SYMBOLS_BATCH = tf.get_collection("QUESTION_SYMBOLS_BATCH")
QUESTION_NUMBERS_BATCH = tf.get_collection("QUESTION_NUMBERS_BATCH")
QUESTION_VECTORS_BATCH = tf.get_collection("QUESTION_VECTORS_BATCH")
QUESTION_CONNECTIONS_BATCH = tf.get_collection("QUESTION_CONNECTIONS_BATCH")
ANSWER_SPAN_BATCH = tf.get_collection("ANSWER_SPAN_BATCH")
LEARNING_RATE = tf.get_collection("LEARNING_RATE")[0]
MODEL_UPDATE = tf.get_collection("MODEL_UPDATE")[0]
MODEL_PREDICTS = tf.get_collection("MODEL_PREDICTS")[0]

with tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
) as SESSION:
    SAVER.restore(sess=SESSION, save_path=model_storage_path)
    model_progress = load_file(model_progress_path, "json")

    while True:
        early_stopping_trigger_count = sum(
            model_progress[index]["f1"] <= max(item["f1"] for item in model_progress[:index])
            for index in range(len(model_progress))
            if index > 0
        )

        if early_stopping_trigger_count > early_stopping_trigger_limit:
            break

        begin_time = datetime.datetime.now()
        learning_rate = learning_rate_annealing_schedule(early_stopping_trigger_count)
        train_queue = random.sample(train_composite, len(train_composite))
        train_queue += random.sample(train_composite, (batch_size - len(train_queue) % batch_size) % batch_size)
        develop_queue = random.sample(develop_composite, len(develop_composite))
        develop_queue += random.sample(develop_composite, (batch_size - len(develop_queue) % batch_size) % batch_size)
        develop_solution = {}

        for batch in [train_queue[index:index + batch_size] for index in range(0, len(train_queue), batch_size)]:
            feed_batch = preload_composite(batch, bert_client)
            feed_dict = {LEARNING_RATE: learning_rate}

            for index in range(batch_size):
                feed_dict[PASSAGE_SYMBOLS_BATCH[index]] = feed_batch[index]["passage_symbols"]
                feed_dict[PASSAGE_NUMBERS_BATCH[index]] = feed_batch[index]["passage_numbers"]
                feed_dict[PASSAGE_VECTORS_BATCH[index]] = feed_batch[index]["passage_vectors"]
                feed_dict[PASSAGE_CONNECTIONS_BATCH[index]] = feed_batch[index]["passage_connections"]
                feed_dict[QUESTION_SYMBOLS_BATCH[index]] = feed_batch[index]["question_symbols"]
                feed_dict[QUESTION_NUMBERS_BATCH[index]] = feed_batch[index]["question_numbers"]
                feed_dict[QUESTION_VECTORS_BATCH[index]] = feed_batch[index]["question_vectors"]
                feed_dict[QUESTION_CONNECTIONS_BATCH[index]] = feed_batch[index]["question_connections"]
                feed_dict[ANSWER_SPAN_BATCH[index]] = feed_batch[index]["answer_span"]

            SESSION.run(fetches=MODEL_UPDATE, feed_dict=feed_dict)

        for batch in [develop_queue[index:index + batch_size] for index in range(0, len(develop_queue), batch_size)]:
            feed_batch = preload_composite(batch, bert_client)
            feed_dict = {}

            for index in range(batch_size):
                feed_dict[PASSAGE_SYMBOLS_BATCH[index]] = feed_batch[index]["passage_symbols"]
                feed_dict[PASSAGE_NUMBERS_BATCH[index]] = feed_batch[index]["passage_numbers"]
                feed_dict[PASSAGE_VECTORS_BATCH[index]] = feed_batch[index]["passage_vectors"]
                feed_dict[PASSAGE_CONNECTIONS_BATCH[index]] = feed_batch[index]["passage_connections"]
                feed_dict[QUESTION_SYMBOLS_BATCH[index]] = feed_batch[index]["question_symbols"]
                feed_dict[QUESTION_NUMBERS_BATCH[index]] = feed_batch[index]["question_numbers"]
                feed_dict[QUESTION_VECTORS_BATCH[index]] = feed_batch[index]["question_vectors"]
                feed_dict[QUESTION_CONNECTIONS_BATCH[index]] = feed_batch[index]["question_connections"]

            model_predicts = SESSION.run(fetches=MODEL_PREDICTS, feed_dict=feed_dict)

            for record, predict in zip(feed_batch, model_predicts):
                develop_solution[record["question_id"]] = spacy_nlp(
                    record["passage_string"]
                )[predict[0]:predict[1] + 1].text

        dump_data(develop_solution, develop_solution_path, "json")

        epoch_result = json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    evaluate_script_path,
                    develop_dataset_path,
                    develop_solution_path
                ]
            )
        )

        if len(model_progress) == 0 or epoch_result["f1"] > max(item["f1"] for item in model_progress):
            SAVER.save(sess=SESSION, save_path=model_storage_path, write_meta_graph=False, write_state=False)
            print("optimize: accept {}".format(json.dumps(epoch_result)))

        else:
            SAVER.restore(sess=SESSION, save_path=model_storage_path)
            print("optimize: cancel {}".format(json.dumps(epoch_result)))

        model_progress.append(epoch_result)
        dump_data(model_progress, model_progress_path, "json")

        print(
            "optimize: cost {} seconds to complete epoch {}".format(
                int((datetime.datetime.now() - begin_time).total_seconds()),
                len(model_progress)
            )
        )

if bert_client is not None:
    bert_client.close()

if bert_server is not None:
    bert_server.close()
