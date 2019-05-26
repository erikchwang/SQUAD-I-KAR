import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse, datetime, json, multiprocessing, nltk, random, spacy, subprocess, sys
import tensorflow as tf, tensorflow_hub as tfh
from nltk.corpus import stopwords, wordnet

gpu_count = 4
batch_size = 32
hop_limit = 3
span_limit = 16
glove_size = 300
lstm_size = 300
dropout_rate = 0.2
gradient_global_norm_limit = 5.0
learning_rate_initial_value = 0.0005
learning_rate_decay_rate = 0.5
learning_rate_decay_limit = 5
ema_manager_decay_rate = 0.999
train_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/train_dataset")
develop_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/develop_dataset")
evaluate_script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/evaluate_script")
glove_archive_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove_archive")
vocabulary_catalog_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vocabulary_catalog")
vocabulary_glove_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vocabulary_glove")
train_composite_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/train_composite")
develop_composite_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/develop_composite")
develop_solution_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/develop_solution")
model_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model/model_graph")
model_storage_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model/model_storage")
model_progress_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model/model_progress")
nltk.data.path = [os.path.join(os.path.dirname(os.path.realpath(__file__)), "nltk")]
elmo_module_url = "https://tfhub.dev/google/elmo/2"
spacy_nlp = spacy.load(name="en_core_web_lg", disable=["parser", "ner"])
stopwords_words = stopwords.words("english")

wordnet_posmap = {
    "NN": wordnet.NOUN, "NNP": wordnet.NOUN, "NNPS": wordnet.NOUN, "NNS": wordnet.NOUN,
    "VB": wordnet.VERB, "VBD": wordnet.VERB, "VBG": wordnet.VERB,
    "VBN": wordnet.VERB, "VBP": wordnet.VERB, "VBZ": wordnet.VERB,
    "JJ": wordnet.ADJ, "JJR": wordnet.ADJ, "JJS": wordnet.ADJ,
    "RB": wordnet.ADV, "RBR": wordnet.ADV, "RBS": wordnet.ADV, "RP": wordnet.ADV
}

wordnet_relations = [
    "hypernyms", "instance_hypernyms",
    "hyponyms", "instance_hyponyms",
    "member_holonyms", "substance_holonyms", "part_holonyms",
    "member_meronyms", "substance_meronyms", "part_meronyms",
    "attributes", "entailments", "causes", "also_sees", "verb_groups", "similar_tos"
]


def load_file(file_path, file_type):
    if file_type == "text":
        with open(file_path, "rt") as file_stream:
            return file_stream.read().splitlines()

    elif file_type == "json":
        with open(file_path, "rt") as file_stream:
            return json.load(file_stream)

    else:
        raise Exception("invalid file type: {}".format(file_type))


def dump_data(data_buffer, file_path, file_type):
    if file_type == "text":
        with open(file_path, "wt") as file_stream:
            file_stream.write("\n".join(data_buffer))

    elif file_type == "json":
        with open(file_path, "wt") as file_stream:
            json.dump(obj=data_buffer, fp=file_stream)

    else:
        raise Exception("invalid file type: {}".format(file_type))


def convert_dataset(dataset_buffer, vocabulary_catalog, require_answer):
    def get_text_symbols(text_tokens):
        return [token.text for token in text_tokens]

    def get_text_numbers(text_tokens):
        return [
            vocabulary_catalog.index(token.text) if token.text in vocabulary_catalog else 0
            for token in text_tokens
        ]

    def get_text_span(text_tokens, span_range):
        for start_index, start_token in enumerate(text_tokens):
            if start_token.idx <= span_range[0] < start_token.idx + len(start_token):
                for end_index, end_token in enumerate(text_tokens[start_index:], start_index):
                    if end_token.idx < span_range[1] <= end_token.idx + len(end_token):
                        return [start_index, end_index]

    composite_records = []

    for article in dataset_buffer["data"]:
        for paragraph in article["paragraphs"]:
            passage_source = paragraph["context"]
            passage_tokens = spacy_nlp(passage_source)
            passage_symbols = get_text_symbols(passage_tokens)
            passage_numbers = get_text_numbers(passage_tokens)

            for qa in paragraph["qas"]:
                question_source = qa["question"]
                question_tokens = spacy_nlp(question_source)
                question_symbols = get_text_symbols(question_tokens)
                question_numbers = get_text_numbers(question_tokens)

                composite_record = {
                    "passage_source": passage_source,
                    "passage_symbols": passage_symbols,
                    "passage_numbers": passage_numbers,
                    "question_source": question_source,
                    "question_symbols": question_symbols,
                    "question_numbers": question_numbers
                }

                if require_answer:
                    for answer in qa["answers"]:
                        answer_span = get_text_span(
                            passage_tokens,
                            [answer["answer_start"], answer["answer_start"] + len(answer["text"])]
                        )

                        if answer_span is not None:
                            composite_record["answer_span"] = answer_span
                            composite_records.append(composite_record.copy())

                else:
                    question_id = qa["id"]
                    composite_record["question_id"] = question_id
                    composite_records.append(composite_record)

    return composite_records


def enrich_composite(composite_record):
    def get_text_nodes(text_tokens):
        text_nodes = []

        for token in text_tokens:
            if token.text.lower() not in stopwords_words and token.tag_ in wordnet_posmap:
                direct_synsets = set(wordnet.synsets(lemma=token.text, pos=wordnet_posmap[token.tag_]))

            else:
                direct_synsets = set()

            spread_synsets = direct_synsets.copy()

            if len(spread_synsets) != 0:
                current_synsets = spread_synsets

                for _ in range(hop_limit):
                    current_synsets = {
                        relative
                        for synset in current_synsets
                        for relation in wordnet_relations
                        for relative in getattr(synset, relation)()
                    }

                    spread_synsets.update(current_synsets)

            text_nodes.append({"direct_synsets": direct_synsets, "spread_synsets": spread_synsets})

        return text_nodes

    def get_text_passage(text_nodes, passage_nodes):
        text_passage_indices, text_passage_values = zip(
            *[
                 ([text_index, passage_index], True)
                 for text_index, text_node in enumerate(text_nodes)
                 for passage_index, passage_node in enumerate(passage_nodes)
                 if passage_node is not text_node and
                    len(passage_node["direct_synsets"].intersection(text_node["spread_synsets"])) != 0
             ] or [([0, 0], False)]
        )

        text_passage_indices = list(text_passage_indices)
        text_passage_values = list(text_passage_values)
        text_passage_shape = [len(text_nodes), len(passage_nodes)]

        return text_passage_indices, text_passage_values, text_passage_shape

    passage_nodes = get_text_nodes(spacy_nlp(composite_record["passage_source"]))
    question_nodes = get_text_nodes(spacy_nlp(composite_record["question_source"]))

    passage_passage_indices, passage_passage_values, passage_passage_shape = get_text_passage(
        passage_nodes,
        passage_nodes
    )

    question_passage_indices, question_passage_values, question_passage_shape = get_text_passage(
        question_nodes,
        passage_nodes
    )

    composite_record["passage_passage_indices"] = passage_passage_indices
    composite_record["passage_passage_values"] = passage_passage_values
    composite_record["passage_passage_shape"] = passage_passage_shape
    composite_record["question_passage_indices"] = question_passage_indices
    composite_record["question_passage_values"] = question_passage_values
    composite_record["question_passage_shape"] = question_passage_shape

    return composite_record


def feed_forward(
        ELMO_MODULE, GLOVE_TABLE,
        PASSAGE_SYMBOLS, QUESTION_SYMBOLS,
        PASSAGE_NUMBERS, QUESTION_NUMBERS,
        PASSAGE_PASSAGE_INDICES, QUESTION_PASSAGE_INDICES,
        PASSAGE_PASSAGE_VALUES, QUESTION_PASSAGE_VALUES,
        PASSAGE_PASSAGE_SHAPE, QUESTION_PASSAGE_SHAPE,
        require_update
):
    def get_elmo_outputs(TARGET_INPUTS):
        return tf.squeeze(
            input=ELMO_MODULE(
                inputs={
                    "tokens": tf.expand_dims(input=TARGET_INPUTS, axis=0),
                    "sequence_len": tf.expand_dims(input=tf.size(TARGET_INPUTS), axis=0)
                },
                signature="tokens",
                as_dict=True
            )["elmo"],
            axis=[0]
        )

    def get_bilstm_outputs(TARGET_INPUTS):
        return tf.squeeze(
            input=tf.concat(
                values=tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=tf.nn.rnn_cell.DropoutWrapper(
                        cell=tf.nn.rnn_cell.LSTMCell(lstm_size),
                        input_keep_prob=1.0 - dropout_rate if require_update else 1.0
                    ),
                    cell_bw=tf.nn.rnn_cell.DropoutWrapper(
                        cell=tf.nn.rnn_cell.LSTMCell(lstm_size),
                        input_keep_prob=1.0 - dropout_rate if require_update else 1.0
                    ),
                    inputs=tf.expand_dims(input=TARGET_INPUTS, axis=0),
                    dtype=tf.float32
                )[0],
                axis=2
            ),
            axis=[0]
        )

    def get_attention_combination(SUBJECT_INPUTS, OBJECT_INPUTS):
        TRANSFORM_GATES = tf.layers.dense(
            inputs=tf.layers.dropout(
                inputs=tf.concat(
                    values=[
                        SUBJECT_INPUTS,
                        OBJECT_INPUTS,
                        tf.multiply(x=SUBJECT_INPUTS, y=OBJECT_INPUTS),
                        tf.subtract(x=SUBJECT_INPUTS, y=OBJECT_INPUTS)
                    ],
                    axis=1
                ),
                rate=dropout_rate,
                training=require_update
            ),
            units=1,
            activation=tf.sigmoid
        )

        TRANSFORM_INFOS = tf.layers.dense(
            inputs=tf.layers.dropout(
                inputs=tf.concat(
                    values=[
                        SUBJECT_INPUTS,
                        OBJECT_INPUTS,
                        tf.multiply(x=SUBJECT_INPUTS, y=OBJECT_INPUTS),
                        tf.subtract(x=SUBJECT_INPUTS, y=OBJECT_INPUTS)
                    ],
                    axis=1
                ),
                rate=dropout_rate,
                training=require_update
            ),
            units=SUBJECT_INPUTS.shape.as_list()[1],
            activation=lambda TARGET_INPUTS: tf.multiply(
                x=tf.multiply(x=TARGET_INPUTS, y=0.5),
                y=tf.add(x=tf.erf(tf.divide(x=TARGET_INPUTS, y=tf.sqrt(2.0))), y=1.0)
            )
        )

        return tf.add(
            x=tf.multiply(x=TRANSFORM_GATES, y=TRANSFORM_INFOS),
            y=tf.multiply(x=tf.subtract(x=1.0, y=TRANSFORM_GATES), y=SUBJECT_INPUTS)
        )

    def get_knowledge_normalization(TARGET_SIMILARITY, TARGET_CONNECTION):
        return tf.multiply(
            x=tf.nn.softmax(
                tf.where(
                    condition=TARGET_CONNECTION,
                    x=TARGET_SIMILARITY,
                    y=tf.where(
                        condition=tf.reduce_any(input_tensor=TARGET_CONNECTION, axis=1),
                        x=tf.fill(dims=tf.shape(TARGET_CONNECTION), value=-float("inf")),
                        y=tf.fill(dims=tf.shape(TARGET_CONNECTION), value=0.0)
                    )
                )
            ),
            y=tf.cast(x=TARGET_CONNECTION, dtype=tf.float32)
        )

    def get_attention_similarity(SUBJECT_INPUTS, OBJECT_INPUTS):
        SUBJECT_WEIGHT = tf.get_variable(name="SUBJECT_WEIGHT", shape=[1, SUBJECT_INPUTS.shape.as_list()[1]])
        OBJECT_WEIGHT = tf.get_variable(name="OBJECT_WEIGHT", shape=[1, SUBJECT_INPUTS.shape.as_list()[1]])
        PRODUCT_WEIGHT = tf.get_variable(name="PRODUCT_WEIGHT", shape=[1, SUBJECT_INPUTS.shape.as_list()[1]])
        DIFFERENCE_WEIGHT = tf.get_variable(name="DIFFERENCE_WEIGHT", shape=[1, SUBJECT_INPUTS.shape.as_list()[1]])

        return tf.add_n(
            [
                tf.tile(
                    input=tf.matmul(
                        a=tf.layers.dropout(inputs=SUBJECT_INPUTS, rate=dropout_rate, training=require_update),
                        b=SUBJECT_WEIGHT,
                        transpose_b=True
                    ),
                    multiples=[1, tf.shape(OBJECT_INPUTS)[0]]
                ),
                tf.tile(
                    input=tf.matmul(
                        a=OBJECT_WEIGHT,
                        b=tf.layers.dropout(inputs=OBJECT_INPUTS, rate=dropout_rate, training=require_update),
                        transpose_b=True
                    ),
                    multiples=[tf.shape(SUBJECT_INPUTS)[0], 1]
                ),
                tf.matmul(
                    a=tf.multiply(
                        x=tf.layers.dropout(inputs=SUBJECT_INPUTS, rate=dropout_rate, training=require_update),
                        y=PRODUCT_WEIGHT
                    ),
                    b=tf.layers.dropout(inputs=OBJECT_INPUTS, rate=dropout_rate, training=require_update),
                    transpose_b=True
                ),
                tf.subtract(
                    x=tf.matmul(
                        a=tf.layers.dropout(inputs=SUBJECT_INPUTS, rate=dropout_rate, training=require_update),
                        b=DIFFERENCE_WEIGHT,
                        transpose_b=True
                    ),
                    y=tf.matmul(
                        a=DIFFERENCE_WEIGHT,
                        b=tf.layers.dropout(inputs=OBJECT_INPUTS, rate=dropout_rate, training=require_update),
                        transpose_b=True
                    )
                )
            ]
        )

    def get_attention_distribution(SUBJECT_INPUT, OBJECT_INPUTS):
        return tf.matmul(
            a=tf.layers.dense(
                inputs=tf.layers.dropout(inputs=SUBJECT_INPUT, rate=dropout_rate, training=require_update),
                units=OBJECT_INPUTS.shape.as_list()[1],
                use_bias=False
            ),
            b=tf.layers.dropout(inputs=OBJECT_INPUTS, rate=dropout_rate, training=require_update),
            transpose_b=True
        ) if SUBJECT_INPUT is not None else tf.transpose(
            tf.layers.dense(
                inputs=tf.layers.dropout(inputs=OBJECT_INPUTS, rate=dropout_rate, training=require_update),
                units=1,
                use_bias=False
            )
        )

    PASSAGE_ELMO_CODES = get_elmo_outputs(PASSAGE_SYMBOLS)
    QUESTION_ELMO_CODES = get_elmo_outputs(QUESTION_SYMBOLS)
    PASSAGE_GLOVE_CODES = tf.gather(params=GLOVE_TABLE, indices=PASSAGE_NUMBERS)
    QUESTION_GLOVE_CODES = tf.gather(params=GLOVE_TABLE, indices=QUESTION_NUMBERS)

    PASSAGE_PASSAGE_CONNECTION = tf.scatter_nd(
        indices=PASSAGE_PASSAGE_INDICES,
        updates=PASSAGE_PASSAGE_VALUES,
        shape=PASSAGE_PASSAGE_SHAPE
    )

    QUESTION_PASSAGE_CONNECTION = tf.scatter_nd(
        indices=QUESTION_PASSAGE_INDICES,
        updates=QUESTION_PASSAGE_VALUES,
        shape=QUESTION_PASSAGE_SHAPE
    )

    with tf.variable_scope("CONTEXT"):
        with tf.variable_scope(name_or_scope="CONTEXT", reuse=None):
            PASSAGE_CONTEXT_CODES = tf.concat(
                values=[
                    PASSAGE_ELMO_CODES,
                    get_bilstm_outputs(tf.concat(values=[PASSAGE_ELMO_CODES, PASSAGE_GLOVE_CODES], axis=1))
                ],
                axis=1
            )

            PASSAGE_CONTEXT_KEYS = get_attention_combination(
                PASSAGE_CONTEXT_CODES,
                tf.matmul(
                    a=get_knowledge_normalization(
                        get_attention_similarity(PASSAGE_CONTEXT_CODES, PASSAGE_CONTEXT_CODES),
                        PASSAGE_PASSAGE_CONNECTION
                    ),
                    b=PASSAGE_CONTEXT_CODES
                )
            )

        with tf.variable_scope(name_or_scope="CONTEXT", reuse=True):
            QUESTION_CONTEXT_CODES = tf.concat(
                values=[
                    QUESTION_ELMO_CODES,
                    get_bilstm_outputs(tf.concat(values=[QUESTION_ELMO_CODES, QUESTION_GLOVE_CODES], axis=1))
                ],
                axis=1
            )

            QUESTION_CONTEXT_KEYS = get_attention_combination(
                QUESTION_CONTEXT_CODES,
                tf.matmul(
                    a=get_knowledge_normalization(
                        get_attention_similarity(QUESTION_CONTEXT_CODES, PASSAGE_CONTEXT_CODES),
                        QUESTION_PASSAGE_CONNECTION
                    ),
                    b=PASSAGE_CONTEXT_CODES
                )
            )

    with tf.variable_scope("MEMORY"):
        with tf.variable_scope("SIMILARITY"):
            PASSAGE_QUESTION_SIMILARITY = get_attention_similarity(PASSAGE_CONTEXT_KEYS, QUESTION_CONTEXT_KEYS)
            QUESTION_PASSAGE_SIMILARITY = tf.transpose(PASSAGE_QUESTION_SIMILARITY)

        with tf.variable_scope(name_or_scope="MEMORY", reuse=None):
            PASSAGE_MEMORY_CODES = get_attention_combination(
                PASSAGE_CONTEXT_CODES,
                tf.matmul(a=tf.nn.softmax(PASSAGE_QUESTION_SIMILARITY), b=QUESTION_CONTEXT_CODES)
            )

        with tf.variable_scope(name_or_scope="MEMORY", reuse=True):
            QUESTION_MEMORY_CODES = get_attention_combination(
                QUESTION_CONTEXT_CODES,
                tf.matmul(a=tf.nn.softmax(QUESTION_PASSAGE_SIMILARITY), b=PASSAGE_CONTEXT_CODES)
            )

        with tf.variable_scope("PASSAGE"):
            PASSAGE_MEMORY_CODES = get_bilstm_outputs(PASSAGE_MEMORY_CODES)

            PASSAGE_MEMORY_KEYS = get_attention_combination(
                PASSAGE_MEMORY_CODES,
                tf.matmul(
                    a=get_knowledge_normalization(
                        get_attention_similarity(PASSAGE_MEMORY_CODES, PASSAGE_MEMORY_CODES),
                        PASSAGE_PASSAGE_CONNECTION
                    ),
                    b=PASSAGE_MEMORY_CODES
                )
            )

        with tf.variable_scope("QUESTION"):
            QUESTION_MEMORY_CODES = get_bilstm_outputs(QUESTION_MEMORY_CODES)

    with tf.variable_scope("SUMMARY"):
        with tf.variable_scope("SIMILARITY"):
            PASSAGE_PASSAGE_SIMILARITY = tf.matrix_set_diag(
                input=get_attention_similarity(PASSAGE_MEMORY_KEYS, PASSAGE_MEMORY_KEYS),
                diagonal=tf.fill(dims=[tf.shape(PASSAGE_MEMORY_KEYS)[0]], value=-float("inf"))
            )

        with tf.variable_scope("PASSAGE"):
            PASSAGE_SUMMARY_CODES = get_bilstm_outputs(
                get_attention_combination(
                    PASSAGE_MEMORY_CODES,
                    tf.matmul(a=tf.nn.softmax(PASSAGE_PASSAGE_SIMILARITY), b=PASSAGE_MEMORY_CODES)
                )
            )

        with tf.variable_scope("QUESTION"):
            QUESTION_SUMMARY_CODE = tf.matmul(
                a=tf.nn.softmax(get_attention_distribution(None, QUESTION_MEMORY_CODES)),
                b=QUESTION_MEMORY_CODES
            )

    with tf.variable_scope("OUTPUT"):
        ANSWER_SPAN_DISTRIBUTION = tf.concat(
            values=[
                get_attention_distribution(QUESTION_SUMMARY_CODE, PASSAGE_SUMMARY_CODES),
                get_attention_distribution(QUESTION_SUMMARY_CODE, PASSAGE_SUMMARY_CODES)
            ],
            axis=0
        )

        return ANSWER_SPAN_DISTRIBUTION


def build_update(
        ELMO_MODULE, GLOVE_TABLE,
        PASSAGE_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
        PASSAGE_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
        PASSAGE_PASSAGE_INDICES_BATCH, QUESTION_PASSAGE_INDICES_BATCH,
        PASSAGE_PASSAGE_VALUES_BATCH, QUESTION_PASSAGE_VALUES_BATCH,
        PASSAGE_PASSAGE_SHAPE_BATCH, QUESTION_PASSAGE_SHAPE_BATCH,
        ANSWER_SPAN_BATCH, LEARNING_RATE, EMA_MANAGER
):
    def get_greedy_sample(TARGET_DISTRIBUTION):
        INDEX_SAMPLE = tf.cast(
            x=tf.argmax(
                tf.reshape(
                    tensor=tf.matrix_band_part(
                        input=tf.matmul(
                            a=tf.nn.softmax(TARGET_DISTRIBUTION[:1]),
                            b=tf.nn.softmax(TARGET_DISTRIBUTION[1:]),
                            transpose_a=True
                        ),
                        num_lower=tf.cast(x=0, dtype=tf.int32),
                        num_upper=tf.subtract(x=tf.minimum(x=tf.shape(TARGET_DISTRIBUTION)[1], y=span_limit), y=1)
                    ),
                    shape=[-1]
                )
            ),
            dtype=tf.int32
        )

        return tf.stack(
            [
                tf.floordiv(x=INDEX_SAMPLE, y=tf.shape(TARGET_DISTRIBUTION)[1]),
                tf.floormod(x=INDEX_SAMPLE, y=tf.shape(TARGET_DISTRIBUTION)[1])
            ]
        )

    def get_random_sample(TARGET_DISTRIBUTION):
        INDEX_SAMPLE = tf.squeeze(
            tf.cast(
                x=tf.multinomial(
                    logits=tf.reshape(
                        tensor=tf.where(
                            condition=tf.logical_and(
                                x=tf.greater_equal(
                                    x=tf.subtract(
                                        x=tf.add(
                                            x=tf.expand_dims(input=tf.range(tf.shape(TARGET_DISTRIBUTION)[1]), axis=0),
                                            y=1
                                        ),
                                        y=tf.expand_dims(input=tf.range(tf.shape(TARGET_DISTRIBUTION)[1]), axis=1)
                                    ),
                                    y=1
                                ),
                                y=tf.less_equal(
                                    x=tf.subtract(
                                        x=tf.add(
                                            x=tf.expand_dims(input=tf.range(tf.shape(TARGET_DISTRIBUTION)[1]), axis=0),
                                            y=1
                                        ),
                                        y=tf.expand_dims(input=tf.range(tf.shape(TARGET_DISTRIBUTION)[1]), axis=1)
                                    ),
                                    y=span_limit
                                )
                            ),
                            x=tf.add(x=tf.transpose(TARGET_DISTRIBUTION[:1]), y=TARGET_DISTRIBUTION[1:]),
                            y=tf.fill(
                                dims=[tf.shape(TARGET_DISTRIBUTION)[1], tf.shape(TARGET_DISTRIBUTION)[1]],
                                value=-float("inf")
                            )
                        ),
                        shape=[1, -1]
                    ),
                    num_samples=1
                ),
                dtype=tf.int32
            )
        )

        return tf.stack(
            [
                tf.floordiv(x=INDEX_SAMPLE, y=tf.shape(TARGET_DISTRIBUTION)[1]),
                tf.floormod(x=INDEX_SAMPLE, y=tf.shape(TARGET_DISTRIBUTION)[1])
            ]
        )

    def get_sample_score(TARGET_SAMPLE, TARGET_TRUTH):
        OVERLAP_SIZE = tf.cast(
            x=tf.maximum(
                x=tf.subtract(
                    x=tf.add(x=tf.minimum(x=TARGET_SAMPLE[1], y=TARGET_TRUTH[1]), y=1),
                    y=tf.maximum(x=TARGET_SAMPLE[0], y=TARGET_TRUTH[0])
                ),
                y=0
            ),
            dtype=tf.float32
        )

        PRECISION_SCORE = tf.divide(
            x=OVERLAP_SIZE,
            y=tf.cast(x=tf.subtract(x=tf.add(x=TARGET_SAMPLE[1], y=1), y=TARGET_SAMPLE[0]), dtype=tf.float32)
        )

        RECALL_SCORE = tf.divide(
            x=OVERLAP_SIZE,
            y=tf.cast(x=tf.subtract(x=tf.add(x=TARGET_TRUTH[1], y=1), y=TARGET_TRUTH[0]), dtype=tf.float32)
        )

        return tf.div_no_nan(
            x=tf.multiply(x=tf.multiply(x=PRECISION_SCORE, y=RECALL_SCORE), y=2.0),
            y=tf.add(x=PRECISION_SCORE, y=RECALL_SCORE)
        )

    def get_joint_loss(TARGET_LOSSES, TARGET_VARIANCES):
        return tf.add(
            x=tf.reduce_mean(tf.divide(x=TARGET_LOSSES, y=tf.square(TARGET_VARIANCES))),
            y=tf.reduce_sum(tf.log(tf.square(TARGET_VARIANCES)))
        )

    def get_raw_gradients(TARGET_LOSS, TARGET_VARIABLES):
        return [
            GRADIENT if GRADIENT is not None else tf.zeros_like(VARIABLE)
            for GRADIENT, VARIABLE in zip(tf.gradients(ys=TARGET_LOSS, xs=TARGET_VARIABLES), TARGET_VARIABLES)
        ]

    COARSE_GRADIENTS_BATCH = []
    FINE_GRADIENTS_BATCH = []

    for index in range(batch_size):
        with tf.device("/device:GPU:{}".format(index % gpu_count)):
            with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=None if index == 0 else True):
                ANSWER_SPAN_DISTRIBUTION = feed_forward(
                    ELMO_MODULE, GLOVE_TABLE,
                    PASSAGE_SYMBOLS_BATCH[index], QUESTION_SYMBOLS_BATCH[index],
                    PASSAGE_NUMBERS_BATCH[index], QUESTION_NUMBERS_BATCH[index],
                    PASSAGE_PASSAGE_INDICES_BATCH[index], QUESTION_PASSAGE_INDICES_BATCH[index],
                    PASSAGE_PASSAGE_VALUES_BATCH[index], QUESTION_PASSAGE_VALUES_BATCH[index],
                    PASSAGE_PASSAGE_SHAPE_BATCH[index], QUESTION_PASSAGE_SHAPE_BATCH[index],
                    True
                )

                GREEDY_SAMPLE_SPAN = get_greedy_sample(ANSWER_SPAN_DISTRIBUTION)
                RANDOM_SAMPLE_SPAN = get_random_sample(ANSWER_SPAN_DISTRIBUTION)
                GREEDY_SAMPLE_SCORE = get_sample_score(GREEDY_SAMPLE_SPAN, ANSWER_SPAN_BATCH[index])
                RANDOM_SAMPLE_SCORE = get_sample_score(RANDOM_SAMPLE_SPAN, ANSWER_SPAN_BATCH[index])

                COARSE_OVERALL_LOSS = tf.losses.sparse_softmax_cross_entropy(
                    labels=ANSWER_SPAN_BATCH[index],
                    logits=ANSWER_SPAN_DISTRIBUTION,
                    reduction=tf.losses.Reduction.SUM
                )

                FINE_OVERALL_LOSS = get_joint_loss(
                    tf.stack(
                        [
                            COARSE_OVERALL_LOSS,
                            tf.losses.sparse_softmax_cross_entropy(
                                labels=tf.stack([GREEDY_SAMPLE_SPAN, RANDOM_SAMPLE_SPAN]),
                                logits=tf.stack([ANSWER_SPAN_DISTRIBUTION, ANSWER_SPAN_DISTRIBUTION]),
                                weights=tf.stop_gradient(
                                    tf.expand_dims(
                                        input=tf.maximum(
                                            x=tf.subtract(
                                                x=tf.stack([GREEDY_SAMPLE_SCORE, RANDOM_SAMPLE_SCORE]),
                                                y=tf.stack([RANDOM_SAMPLE_SCORE, GREEDY_SAMPLE_SCORE])
                                            ),
                                            y=0.0
                                        ),
                                        axis=1
                                    )
                                ),
                                reduction=tf.losses.Reduction.SUM
                            )
                        ]
                    ),
                    tf.get_variable(name="TASK_VARIANCES", shape=[2], initializer=tf.ones_initializer())
                )

                COARSE_GRADIENTS_BATCH.append(get_raw_gradients(COARSE_OVERALL_LOSS, tf.trainable_variables()))
                FINE_GRADIENTS_BATCH.append(get_raw_gradients(FINE_OVERALL_LOSS, tf.trainable_variables()))

    with tf.device("/cpu:0"):
        COARSE_VARIABLES_UPDATE = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(
            zip(
                tf.clip_by_global_norm(
                    t_list=[
                        tf.reduce_mean(input_tensor=tf.stack(BATCH), axis=0)
                        for BATCH in zip(*COARSE_GRADIENTS_BATCH)
                    ],
                    clip_norm=gradient_global_norm_limit
                )[0],
                tf.trainable_variables()
            )
        )

        FINE_VARIABLES_UPDATE = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(
            zip(
                tf.clip_by_global_norm(
                    t_list=[
                        tf.reduce_mean(input_tensor=tf.stack(BATCH), axis=0)
                        for BATCH in zip(*FINE_GRADIENTS_BATCH)
                    ],
                    clip_norm=gradient_global_norm_limit
                )[0],
                tf.trainable_variables()
            )
        )

        EMA_MANAGER_UPDATE = EMA_MANAGER.apply(tf.trainable_variables())

        with tf.control_dependencies([COARSE_VARIABLES_UPDATE]):
            COARSE_UPDATE = tf.group(EMA_MANAGER_UPDATE)

        with tf.control_dependencies([FINE_VARIABLES_UPDATE]):
            FINE_UPDATE = tf.group(EMA_MANAGER_UPDATE)

        return COARSE_UPDATE, FINE_UPDATE


def build_predicts(
        ELMO_MODULE, GLOVE_TABLE,
        PASSAGE_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
        PASSAGE_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
        PASSAGE_PASSAGE_INDICES_BATCH, QUESTION_PASSAGE_INDICES_BATCH,
        PASSAGE_PASSAGE_VALUES_BATCH, QUESTION_PASSAGE_VALUES_BATCH,
        PASSAGE_PASSAGE_SHAPE_BATCH, QUESTION_PASSAGE_SHAPE_BATCH
):
    def get_greedy_sample(TARGET_DISTRIBUTION):
        INDEX_SAMPLE = tf.cast(
            x=tf.argmax(
                tf.reshape(
                    tensor=tf.matrix_band_part(
                        input=tf.matmul(
                            a=tf.nn.softmax(TARGET_DISTRIBUTION[:1]),
                            b=tf.nn.softmax(TARGET_DISTRIBUTION[1:]),
                            transpose_a=True
                        ),
                        num_lower=tf.cast(x=0, dtype=tf.int32),
                        num_upper=tf.subtract(x=tf.minimum(x=tf.shape(TARGET_DISTRIBUTION)[1], y=span_limit), y=1)
                    ),
                    shape=[-1]
                )
            ),
            dtype=tf.int32
        )

        return tf.stack(
            [
                tf.floordiv(x=INDEX_SAMPLE, y=tf.shape(TARGET_DISTRIBUTION)[1]),
                tf.floormod(x=INDEX_SAMPLE, y=tf.shape(TARGET_DISTRIBUTION)[1])
            ]
        )

    MODEL_PREDICT_BATCH = []

    for index in range(batch_size):
        with tf.device("/device:GPU:{}".format(index % gpu_count)):
            with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=True):
                ANSWER_SPAN_DISTRIBUTION = feed_forward(
                    ELMO_MODULE, GLOVE_TABLE,
                    PASSAGE_SYMBOLS_BATCH[index], QUESTION_SYMBOLS_BATCH[index],
                    PASSAGE_NUMBERS_BATCH[index], QUESTION_NUMBERS_BATCH[index],
                    PASSAGE_PASSAGE_INDICES_BATCH[index], QUESTION_PASSAGE_INDICES_BATCH[index],
                    PASSAGE_PASSAGE_VALUES_BATCH[index], QUESTION_PASSAGE_VALUES_BATCH[index],
                    PASSAGE_PASSAGE_SHAPE_BATCH[index], QUESTION_PASSAGE_SHAPE_BATCH[index],
                    False
                )

                MODEL_PREDICT_BATCH.append(get_greedy_sample(ANSWER_SPAN_DISTRIBUTION))

    with tf.device("/cpu:0"):
        MODEL_PREDICTS = tf.stack(MODEL_PREDICT_BATCH)

        return MODEL_PREDICTS


def build_predict(
        ELMO_MODULE, GLOVE_TABLE,
        PASSAGE_SYMBOLS, QUESTION_SYMBOLS,
        PASSAGE_NUMBERS, QUESTION_NUMBERS,
        PASSAGE_PASSAGE_INDICES, QUESTION_PASSAGE_INDICES,
        PASSAGE_PASSAGE_VALUES, QUESTION_PASSAGE_VALUES,
        PASSAGE_PASSAGE_SHAPE, QUESTION_PASSAGE_SHAPE
):
    def get_greedy_sample(TARGET_DISTRIBUTION):
        INDEX_SAMPLE = tf.cast(
            x=tf.argmax(
                tf.reshape(
                    tensor=tf.matrix_band_part(
                        input=tf.matmul(
                            a=tf.nn.softmax(TARGET_DISTRIBUTION[:1]),
                            b=tf.nn.softmax(TARGET_DISTRIBUTION[1:]),
                            transpose_a=True
                        ),
                        num_lower=tf.cast(x=0, dtype=tf.int32),
                        num_upper=tf.subtract(x=tf.minimum(x=tf.shape(TARGET_DISTRIBUTION)[1], y=span_limit), y=1)
                    ),
                    shape=[-1]
                )
            ),
            dtype=tf.int32
        )

        return tf.stack(
            [
                tf.floordiv(x=INDEX_SAMPLE, y=tf.shape(TARGET_DISTRIBUTION)[1]),
                tf.floormod(x=INDEX_SAMPLE, y=tf.shape(TARGET_DISTRIBUTION)[1])
            ]
        )

    with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=True):
        ANSWER_SPAN_DISTRIBUTION = feed_forward(
            ELMO_MODULE, GLOVE_TABLE,
            PASSAGE_SYMBOLS, QUESTION_SYMBOLS,
            PASSAGE_NUMBERS, QUESTION_NUMBERS,
            PASSAGE_PASSAGE_INDICES, QUESTION_PASSAGE_INDICES,
            PASSAGE_PASSAGE_VALUES, QUESTION_PASSAGE_VALUES,
            PASSAGE_PASSAGE_SHAPE, QUESTION_PASSAGE_SHAPE,
            False
        )

        MODEL_PREDICT = get_greedy_sample(ANSWER_SPAN_DISTRIBUTION)

        return MODEL_PREDICT
