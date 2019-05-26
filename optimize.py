from utility import *

train_composite = load_file(train_composite_path, "json")
develop_composite = load_file(develop_composite_path, "json")
SAVER = tf.train.import_meta_graph(model_graph_path)
PASSAGE_SYMBOLS_BATCH = tf.get_collection("PASSAGE_SYMBOLS_BATCH")
PASSAGE_NUMBERS_BATCH = tf.get_collection("PASSAGE_NUMBERS_BATCH")
PASSAGE_PASSAGE_INDICES_BATCH = tf.get_collection("PASSAGE_PASSAGE_INDICES_BATCH")
PASSAGE_PASSAGE_VALUES_BATCH = tf.get_collection("PASSAGE_PASSAGE_VALUES_BATCH")
PASSAGE_PASSAGE_SHAPE_BATCH = tf.get_collection("PASSAGE_PASSAGE_SHAPE_BATCH")
QUESTION_SYMBOLS_BATCH = tf.get_collection("QUESTION_SYMBOLS_BATCH")
QUESTION_NUMBERS_BATCH = tf.get_collection("QUESTION_NUMBERS_BATCH")
QUESTION_PASSAGE_INDICES_BATCH = tf.get_collection("QUESTION_PASSAGE_INDICES_BATCH")
QUESTION_PASSAGE_VALUES_BATCH = tf.get_collection("QUESTION_PASSAGE_VALUES_BATCH")
QUESTION_PASSAGE_SHAPE_BATCH = tf.get_collection("QUESTION_PASSAGE_SHAPE_BATCH")
ANSWER_SPAN_BATCH = tf.get_collection("ANSWER_SPAN_BATCH")
LEARNING_RATE = tf.get_collection("LEARNING_RATE")[0]
COARSE_UPDATE = tf.get_collection("COARSE_UPDATE")[0]
FINE_UPDATE = tf.get_collection("FINE_UPDATE")[0]
MODEL_PREDICTS = tf.get_collection("MODEL_PREDICTS")[0]

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as SESSION:
    SAVER.restore(sess=SESSION, save_path=model_storage_path)
    model_progress = load_file(model_progress_path, "json")

    while True:
        learning_rate_decay_count = sum(
            item["f1"] != max(item["f1"] for item in model_progress[:index + 1])
            for index, item in enumerate(model_progress)
        )

        if learning_rate_decay_count > learning_rate_decay_limit:
            break

        begin_time = datetime.datetime.now()
        train_epoch = random.sample(train_composite, len(train_composite))
        train_epoch += random.sample(train_composite, (batch_size - len(train_epoch) % batch_size) % batch_size)
        learning_rate = learning_rate_initial_value * learning_rate_decay_rate ** learning_rate_decay_count

        for batch in [train_epoch[index:index + batch_size] for index in range(0, len(train_epoch), batch_size)]:
            feed_dict = {LEARNING_RATE: learning_rate}

            for index in range(batch_size):
                feed_dict[PASSAGE_SYMBOLS_BATCH[index]] = batch[index]["passage_symbols"]
                feed_dict[PASSAGE_NUMBERS_BATCH[index]] = batch[index]["passage_numbers"]
                feed_dict[PASSAGE_PASSAGE_INDICES_BATCH[index]] = batch[index]["passage_passage_indices"]
                feed_dict[PASSAGE_PASSAGE_VALUES_BATCH[index]] = batch[index]["passage_passage_values"]
                feed_dict[PASSAGE_PASSAGE_SHAPE_BATCH[index]] = batch[index]["passage_passage_shape"]
                feed_dict[QUESTION_SYMBOLS_BATCH[index]] = batch[index]["question_symbols"]
                feed_dict[QUESTION_NUMBERS_BATCH[index]] = batch[index]["question_numbers"]
                feed_dict[QUESTION_PASSAGE_INDICES_BATCH[index]] = batch[index]["question_passage_indices"]
                feed_dict[QUESTION_PASSAGE_VALUES_BATCH[index]] = batch[index]["question_passage_values"]
                feed_dict[QUESTION_PASSAGE_SHAPE_BATCH[index]] = batch[index]["question_passage_shape"]
                feed_dict[ANSWER_SPAN_BATCH[index]] = batch[index]["answer_span"]

            if learning_rate == learning_rate_initial_value:
                SESSION.run(fetches=COARSE_UPDATE, feed_dict=feed_dict)

            else:
                SESSION.run(fetches=FINE_UPDATE, feed_dict=feed_dict)

        develop_epoch = random.sample(develop_composite, len(develop_composite))
        develop_epoch += random.sample(develop_composite, (batch_size - len(develop_epoch) % batch_size) % batch_size)
        develop_solution = {}

        for batch in [develop_epoch[index:index + batch_size] for index in range(0, len(develop_epoch), batch_size)]:
            feed_dict = {}

            for index in range(batch_size):
                feed_dict[PASSAGE_SYMBOLS_BATCH[index]] = batch[index]["passage_symbols"]
                feed_dict[PASSAGE_NUMBERS_BATCH[index]] = batch[index]["passage_numbers"]
                feed_dict[PASSAGE_PASSAGE_INDICES_BATCH[index]] = batch[index]["passage_passage_indices"]
                feed_dict[PASSAGE_PASSAGE_VALUES_BATCH[index]] = batch[index]["passage_passage_values"]
                feed_dict[PASSAGE_PASSAGE_SHAPE_BATCH[index]] = batch[index]["passage_passage_shape"]
                feed_dict[QUESTION_SYMBOLS_BATCH[index]] = batch[index]["question_symbols"]
                feed_dict[QUESTION_NUMBERS_BATCH[index]] = batch[index]["question_numbers"]
                feed_dict[QUESTION_PASSAGE_INDICES_BATCH[index]] = batch[index]["question_passage_indices"]
                feed_dict[QUESTION_PASSAGE_VALUES_BATCH[index]] = batch[index]["question_passage_values"]
                feed_dict[QUESTION_PASSAGE_SHAPE_BATCH[index]] = batch[index]["question_passage_shape"]

            predicts = SESSION.run(fetches=MODEL_PREDICTS, feed_dict=feed_dict).tolist()

            for record, predict in zip(batch, predicts):
                develop_solution[record["question_id"]] = spacy_nlp(
                    record["passage_source"]
                )[predict[0]:predict[1] + 1].text

        dump_data(develop_solution, develop_solution_path, "json")

        model_progress.append(
            json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        evaluate_script_path,
                        develop_dataset_path,
                        develop_solution_path
                    ]
                )
            )
        )

        dump_data(model_progress, model_progress_path, "json")

        print(
            "optimize: cost {} seconds to achieve {} with learning rate {}".format(
                int((datetime.datetime.now() - begin_time).total_seconds()),
                json.dumps(model_progress[-1]),
                learning_rate
            )
        )

        if model_progress[-1]["f1"] == max(item["f1"] for item in model_progress):
            SAVER.save(sess=SESSION, save_path=model_storage_path, write_meta_graph=False, write_state=False)
            print("optimize: accepted")

        else:
            SAVER.restore(sess=SESSION, save_path=model_storage_path)
            print("optimize: canceled")
