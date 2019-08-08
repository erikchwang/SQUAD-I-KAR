import logging, os, warnings

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ZEROMQ_SOCK_TMP_DIR"] = "/tmp"
warnings.filterwarnings("ignore")

import argparse, datetime, glob, json, multiprocessing, nltk, numpy, pickle, psutil, random, spacy, subprocess, sys
import tensorflow as tf, tensorflow_hub as tfh
from bert_serving import client, server
from nltk.corpus import stopwords, wordnet

gpu_count = 4
batch_size = 32
layer_size = 512
dropout_rate = 0.2
vector_size = 4096
position_limit = 510
wordnet_relation_hop_count = 3
gradient_clipping_global_norm = 3.0
answer_span_length_limit = 16
exponential_moving_average_decay = 0.999
early_stopping_trigger_limit = 3
learning_rate_annealing_schedule = lambda input: 0.0005 * 0.5 ** input
nltk.data.path = [os.path.join(os.path.dirname(os.path.realpath(__file__)), "nltk")]
glove_archive_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "glove")
bert_archive_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bert")
train_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/train_dataset")
develop_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/develop_dataset")
evaluate_script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/evaluate_script")
word_vocabulary_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/word_vocabulary")
word_embedding_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/word_embedding")
train_composite_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/train_composite")
develop_composite_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/develop_composite")
develop_solution_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/develop_solution")
model_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model/model_graph")
model_storage_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model/model_storage")
model_progress_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model/model_progress")
spacy_nlp = spacy.load(name="en_core_web_lg", disable=["parser", "ner"])
elmo_url = "https://tfhub.dev/google/elmo/2"
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

    elif file_type == "pickle":
        with open(file_path, "rb") as file_stream:
            return pickle.load(file_stream)

    else:
        raise Exception("invalid file type: {}".format(file_type))


def dump_data(data_buffer, file_path, file_type):
    if file_type == "text":
        with open(file_path, "wt") as file_stream:
            file_stream.write("\n".join(data_buffer))

    elif file_type == "json":
        with open(file_path, "wt") as file_stream:
            json.dump(obj=data_buffer, fp=file_stream)

    elif file_type == "pickle":
        with open(file_path, "wb") as file_stream:
            pickle.dump(obj=data_buffer, file=file_stream)

    else:
        raise Exception("invalid file type: {}".format(file_type))


def convert_dataset(dataset_buffer, word_vocabulary, require_answer):
    def get_text_normals(text_tokens):
        return [token.norm_ for token in text_tokens]

    def get_text_postags(text_tokens):
        return [token.tag_ for token in text_tokens]

    def get_text_symbols(text_tokens):
        return [token.text for token in text_tokens]

    def get_text_numbers(text_tokens):
        return [word_vocabulary.index(token.text) if token.text in word_vocabulary else 0 for token in text_tokens]

    def get_text_span(text_tokens, span_offset, span_string):
        span_candidates = [
            {"bound": [start_index, end_index], "error": abs(text_tokens[start_index].idx - span_offset)}
            for start_index in range(len(text_tokens))
            if 0 <= text_tokens[start_index:].text.find(span_string) < len(text_tokens[start_index])
            for end_index in range(start_index, len(text_tokens))
            if span_string not in text_tokens[start_index:end_index].text and
               span_string in text_tokens[start_index:end_index + 1].text
        ]

        if len(span_candidates) > 0:
            return sorted(span_candidates, key=lambda input: input["error"])[0]["bound"]

    composite_buffer = []

    for article in dataset_buffer["data"]:
        for paragraph in article["paragraphs"]:
            passage_string = " ".join(paragraph["context"].split())
            passage_tokens = spacy_nlp(passage_string)[:position_limit]
            passage_normals = get_text_normals(passage_tokens)
            passage_postags = get_text_postags(passage_tokens)
            passage_symbols = get_text_symbols(passage_tokens)
            passage_numbers = get_text_numbers(passage_tokens)

            for qa in paragraph["qas"]:
                question_string = " ".join(qa["question"].split())
                question_tokens = spacy_nlp(question_string)[:position_limit]
                question_normals = get_text_normals(question_tokens)
                question_postags = get_text_postags(question_tokens)
                question_symbols = get_text_symbols(question_tokens)
                question_numbers = get_text_numbers(question_tokens)

                composite_record = {
                    "passage_string": passage_string,
                    "passage_normals": passage_normals,
                    "passage_postags": passage_postags,
                    "passage_symbols": passage_symbols,
                    "passage_numbers": passage_numbers,
                    "question_string": question_string,
                    "question_normals": question_normals,
                    "question_postags": question_postags,
                    "question_symbols": question_symbols,
                    "question_numbers": question_numbers
                }

                if require_answer:
                    for answer in qa["answers"]:
                        answer_span = get_text_span(
                            passage_tokens,
                            answer["answer_start"],
                            " ".join(answer["text"].split())
                        )

                        if answer_span is not None:
                            composite_record["answer_span"] = answer_span
                            composite_buffer.append(composite_record.copy())

                else:
                    question_id = qa["id"]
                    composite_record["question_id"] = question_id
                    composite_buffer.append(composite_record)

    return composite_buffer


def enrich_composite(composite_record):
    def get_text_nodes(text_normals, text_postags):
        text_nodes = []

        for normal, postag in zip(text_normals, text_postags):
            direct_synsets = set()

            if normal not in stopwords_words and postag in wordnet_posmap:
                direct_synsets.update(wordnet.synsets(lemma=normal, pos=wordnet_posmap[postag]))

            spread_synsets = direct_synsets.copy()

            if len(spread_synsets) > 0:
                current_synsets = spread_synsets

                for _ in range(wordnet_relation_hop_count):
                    current_synsets = set(
                        target
                        for synset in current_synsets
                        for relation in wordnet_relations
                        for target in getattr(synset, relation)()
                    )

                    spread_synsets.update(current_synsets)

            text_nodes.append({"direct_synsets": direct_synsets, "spread_synsets": spread_synsets})

        return text_nodes

    def get_text_connections(subject_nodes, object_nodes):
        return numpy.asarray(
            a=[
                  [subject_index, object_index]
                  for subject_index, subject_node in enumerate(subject_nodes)
                  for object_index, object_node in enumerate(object_nodes)
                  if subject_node is not object_node and
                     len(subject_node["spread_synsets"].intersection(object_node["direct_synsets"])) > 0
              ] or numpy.empty(shape=[0, 2], dtype=numpy.int),
            dtype=numpy.int
        )

    composite_record = composite_record.copy()
    passage_nodes = get_text_nodes(composite_record["passage_normals"], composite_record["passage_postags"])
    question_nodes = get_text_nodes(composite_record["question_normals"], composite_record["question_postags"])
    passage_connections = get_text_connections(passage_nodes, passage_nodes)
    question_connections = get_text_connections(question_nodes, passage_nodes)
    composite_record["passage_connections"] = passage_connections
    composite_record["question_connections"] = question_connections

    return composite_record


def preload_composite(composite_buffer, bert_client):
    composite_buffer = [record.copy() for record in composite_buffer]

    if bert_client is not None:
        bert_inputs = [
            item
            for record in composite_buffer
            for item in [
                [symbol.lower() for symbol in record["passage_symbols"]],
                [symbol.lower() for symbol in record["question_symbols"]]
            ]
        ]

        bert_outputs = [
            output[1:len(input) + 1]
            for output, input in zip(bert_client.encode(texts=bert_inputs, is_tokenized=True), bert_inputs)
        ]

        for index, record in enumerate(composite_buffer):
            passage_vectors = bert_outputs[index * 2]
            question_vectors = bert_outputs[index * 2 + 1]
            record["passage_vectors"] = passage_vectors
            record["question_vectors"] = question_vectors

    else:
        for record in composite_buffer:
            passage_vectors = [[]] * len(record["passage_symbols"])
            question_vectors = [[]] * len(record["question_symbols"])
            record["passage_vectors"] = passage_vectors
            record["question_vectors"] = question_vectors

    return composite_buffer


def feed_forward(
        ELMO_MODULE, WORD_EMBEDDING,
        PASSAGE_SYMBOLS, QUESTION_SYMBOLS,
        PASSAGE_NUMBERS, QUESTION_NUMBERS,
        PASSAGE_VECTORS, QUESTION_VECTORS,
        PASSAGE_CONNECTIONS, QUESTION_CONNECTIONS,
        require_update
):
    def get_bilstm_outputs(TARGET_INPUTS):
        return tf.squeeze(
            input=tf.concat(
                values=tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=tf.nn.rnn_cell.DropoutWrapper(
                        cell=tf.nn.rnn_cell.LSTMCell(TARGET_INPUTS.shape.as_list()[1] // 2),
                        input_keep_prob=1.0 - dropout_rate if require_update else 1.0
                    ),
                    cell_bw=tf.nn.rnn_cell.DropoutWrapper(
                        cell=tf.nn.rnn_cell.LSTMCell(TARGET_INPUTS.shape.as_list()[1] // 2),
                        input_keep_prob=1.0 - dropout_rate if require_update else 1.0
                    ),
                    inputs=tf.expand_dims(input=TARGET_INPUTS, axis=0),
                    dtype=tf.float32
                )[0],
                axis=2
            ),
            axis=[0]
        )

    def get_ffnn_outputs(TARGET_INPUTS):
        return tf.layers.dense(
            inputs=tf.layers.dropout(inputs=TARGET_INPUTS, rate=dropout_rate, training=require_update),
            units=layer_size,
            activation=lambda INPUTS: tf.math.multiply(
                x=tf.math.divide(x=INPUTS, y=2.0),
                y=tf.math.add(x=tf.math.erf(tf.math.divide(x=INPUTS, y=2.0 ** 0.5)), y=1.0)
            )
        )

    def get_elmo_outputs(TARGET_INPUTS):
        return tf.squeeze(
            input=ELMO_MODULE(
                inputs={
                    "tokens": tf.expand_dims(input=TARGET_INPUTS, axis=0),
                    "sequence_len": tf.expand_dims(input=tf.size(TARGET_INPUTS), axis=0)
                },
                signature="tokens",
                as_dict=True
            )["word_emb"],
            axis=[0]
        )

    def get_attention_similarity(SUBJECT_INPUTS, OBJECT_INPUTS):
        SUBJECT_WEIGHT = tf.get_variable(name="SUBJECT_WEIGHT", shape=[1, SUBJECT_INPUTS.shape.as_list()[1]])
        OBJECT_WEIGHT = tf.get_variable(name="OBJECT_WEIGHT", shape=[1, SUBJECT_INPUTS.shape.as_list()[1]])
        PRODUCT_WEIGHT = tf.get_variable(name="PRODUCT_WEIGHT", shape=[1, SUBJECT_INPUTS.shape.as_list()[1]])
        DIFFERENCE_WEIGHT = tf.get_variable(name="DIFFERENCE_WEIGHT", shape=[1, SUBJECT_INPUTS.shape.as_list()[1]])

        return tf.math.add_n(
            [
                tf.broadcast_to(
                    input=tf.linalg.matmul(a=SUBJECT_INPUTS, b=SUBJECT_WEIGHT, transpose_b=True),
                    shape=[tf.shape(SUBJECT_INPUTS)[0], tf.shape(OBJECT_INPUTS)[0]]
                ),
                tf.broadcast_to(
                    input=tf.linalg.matmul(a=OBJECT_WEIGHT, b=OBJECT_INPUTS, transpose_b=True),
                    shape=[tf.shape(SUBJECT_INPUTS)[0], tf.shape(OBJECT_INPUTS)[0]]
                ),
                tf.linalg.matmul(
                    a=tf.math.multiply(x=SUBJECT_INPUTS, y=PRODUCT_WEIGHT),
                    b=OBJECT_INPUTS,
                    transpose_b=True
                ),
                tf.math.subtract(
                    x=tf.linalg.matmul(a=SUBJECT_INPUTS, b=DIFFERENCE_WEIGHT, transpose_b=True),
                    y=tf.linalg.matmul(a=DIFFERENCE_WEIGHT, b=OBJECT_INPUTS, transpose_b=True)
                )
            ]
        )

    def get_attention_combination(SUBJECT_INPUTS, OBJECT_INPUTS):
        TRANSFORM_GATES = tf.layers.dense(
            inputs=tf.concat(
                values=[
                    SUBJECT_INPUTS,
                    OBJECT_INPUTS,
                    tf.math.multiply(x=SUBJECT_INPUTS, y=OBJECT_INPUTS),
                    tf.math.subtract(x=SUBJECT_INPUTS, y=OBJECT_INPUTS)
                ],
                axis=1
            ),
            units=1,
            activation=tf.math.sigmoid
        )

        TRANSFORM_INFOS = tf.layers.dense(
            inputs=tf.concat(
                values=[
                    SUBJECT_INPUTS,
                    OBJECT_INPUTS,
                    tf.math.multiply(x=SUBJECT_INPUTS, y=OBJECT_INPUTS),
                    tf.math.subtract(x=SUBJECT_INPUTS, y=OBJECT_INPUTS)
                ],
                axis=1
            ),
            units=SUBJECT_INPUTS.shape.as_list()[1],
            activation=lambda INPUTS: tf.math.multiply(
                x=tf.math.divide(x=INPUTS, y=2.0),
                y=tf.math.add(x=tf.math.erf(tf.math.divide(x=INPUTS, y=2.0 ** 0.5)), y=1.0)
            )
        )

        return tf.math.add(
            x=tf.math.multiply(x=TRANSFORM_GATES, y=TRANSFORM_INFOS),
            y=tf.math.multiply(x=tf.math.subtract(x=1.0, y=TRANSFORM_GATES), y=SUBJECT_INPUTS)
        )

    def get_attention_distribution(SUBJECT_INPUT, OBJECT_INPUTS):
        return tf.linalg.matmul(
            a=tf.layers.dense(inputs=SUBJECT_INPUT, units=OBJECT_INPUTS.shape.as_list()[1], use_bias=False),
            b=OBJECT_INPUTS,
            transpose_b=True
        ) if SUBJECT_INPUT is not None else tf.transpose(tf.layers.dense(inputs=OBJECT_INPUTS, units=1, use_bias=False))

    with tf.variable_scope("CONTEXT"):
        with tf.variable_scope(name_or_scope="CONTEXT", reuse=None):
            PASSAGE_CONTEXT_CODES = get_bilstm_outputs(
                get_ffnn_outputs(
                    tf.concat(
                        values=[
                            get_elmo_outputs(PASSAGE_SYMBOLS),
                            tf.gather(params=WORD_EMBEDDING, indices=PASSAGE_NUMBERS),
                            PASSAGE_VECTORS
                        ],
                        axis=1
                    )
                )
            )

            PASSAGE_CONTEXT_KEYS = get_attention_combination(
                PASSAGE_CONTEXT_CODES,
                tf.sparse.sparse_dense_matmul(
                    sp_a=tf.sparse.softmax(
                        tf.sparse.SparseTensor(
                            indices=tf.dtypes.cast(x=PASSAGE_CONNECTIONS, dtype=tf.int64),
                            values=tf.gather_nd(
                                params=get_attention_similarity(PASSAGE_CONTEXT_CODES, PASSAGE_CONTEXT_CODES),
                                indices=PASSAGE_CONNECTIONS
                            ),
                            dense_shape=[tf.shape(PASSAGE_CONTEXT_CODES)[0], tf.shape(PASSAGE_CONTEXT_CODES)[0]]
                        )
                    ),
                    b=PASSAGE_CONTEXT_CODES
                )
            )

        with tf.variable_scope(name_or_scope="CONTEXT", reuse=True):
            QUESTION_CONTEXT_CODES = get_bilstm_outputs(
                get_ffnn_outputs(
                    tf.concat(
                        values=[
                            get_elmo_outputs(QUESTION_SYMBOLS),
                            tf.gather(params=WORD_EMBEDDING, indices=QUESTION_NUMBERS),
                            QUESTION_VECTORS
                        ],
                        axis=1
                    )
                )
            )

            QUESTION_CONTEXT_KEYS = get_attention_combination(
                QUESTION_CONTEXT_CODES,
                tf.sparse.sparse_dense_matmul(
                    sp_a=tf.sparse.softmax(
                        tf.sparse.SparseTensor(
                            indices=tf.dtypes.cast(x=QUESTION_CONNECTIONS, dtype=tf.int64),
                            values=tf.gather_nd(
                                params=get_attention_similarity(QUESTION_CONTEXT_CODES, PASSAGE_CONTEXT_CODES),
                                indices=QUESTION_CONNECTIONS
                            ),
                            dense_shape=[tf.shape(QUESTION_CONTEXT_CODES)[0], tf.shape(PASSAGE_CONTEXT_CODES)[0]]
                        )
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
                tf.linalg.matmul(a=tf.nn.softmax(PASSAGE_QUESTION_SIMILARITY), b=QUESTION_CONTEXT_CODES)
            )

        with tf.variable_scope(name_or_scope="MEMORY", reuse=True):
            QUESTION_MEMORY_CODES = get_attention_combination(
                QUESTION_CONTEXT_CODES,
                tf.linalg.matmul(a=tf.nn.softmax(QUESTION_PASSAGE_SIMILARITY), b=PASSAGE_CONTEXT_CODES)
            )

        with tf.variable_scope("PASSAGE"):
            PASSAGE_MEMORY_CODES = get_bilstm_outputs(PASSAGE_MEMORY_CODES)

        with tf.variable_scope("QUESTION"):
            QUESTION_MEMORY_CODES = get_bilstm_outputs(QUESTION_MEMORY_CODES)

    with tf.variable_scope("SUMMARY"):
        with tf.variable_scope("SIMILARITY"):
            PASSAGE_PASSAGE_SIMILARITY = tf.sparse.SparseTensor(
                indices=tf.dtypes.cast(x=PASSAGE_CONNECTIONS, dtype=tf.int64),
                values=tf.gather_nd(
                    params=get_attention_similarity(PASSAGE_MEMORY_CODES, PASSAGE_MEMORY_CODES),
                    indices=PASSAGE_CONNECTIONS
                ),
                dense_shape=[tf.shape(PASSAGE_MEMORY_CODES)[0], tf.shape(PASSAGE_MEMORY_CODES)[0]]
            )

        with tf.variable_scope("PASSAGE"):
            PASSAGE_SUMMARY_CODES = get_bilstm_outputs(
                get_attention_combination(
                    PASSAGE_MEMORY_CODES,
                    tf.sparse.sparse_dense_matmul(
                        sp_a=tf.sparse.softmax(PASSAGE_PASSAGE_SIMILARITY),
                        b=PASSAGE_MEMORY_CODES
                    )
                )
            )

        with tf.variable_scope("QUESTION"):
            QUESTION_SUMMARY_CODE = tf.linalg.matmul(
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
        ELMO_MODULE, WORD_EMBEDDING,
        PASSAGE_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
        PASSAGE_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
        PASSAGE_VECTORS_BATCH, QUESTION_VECTORS_BATCH,
        PASSAGE_CONNECTIONS_BATCH, QUESTION_CONNECTIONS_BATCH,
        ANSWER_SPAN_BATCH, LEARNING_RATE, EMA_MANAGER
):
    GRADIENTS_BATCH = []

    for index in range(batch_size):
        with tf.device("/device:GPU:{}".format(index % gpu_count)):
            with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=True if index > 0 else None):
                ANSWER_SPAN_DISTRIBUTION = feed_forward(
                    ELMO_MODULE, WORD_EMBEDDING,
                    PASSAGE_SYMBOLS_BATCH[index], QUESTION_SYMBOLS_BATCH[index],
                    PASSAGE_NUMBERS_BATCH[index], QUESTION_NUMBERS_BATCH[index],
                    PASSAGE_VECTORS_BATCH[index], QUESTION_VECTORS_BATCH[index],
                    PASSAGE_CONNECTIONS_BATCH[index], QUESTION_CONNECTIONS_BATCH[index],
                    True
                )

                GRADIENTS_BATCH.append(
                    tf.gradients(
                        ys=tf.losses.sparse_softmax_cross_entropy(
                            labels=ANSWER_SPAN_BATCH[index],
                            logits=ANSWER_SPAN_DISTRIBUTION,
                            reduction=tf.losses.Reduction.SUM
                        ),
                        xs=tf.trainable_variables()
                    )
                )

    with tf.device("/cpu:0"):
        VARIABLES_UPDATE = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(
            zip(
                tf.clip_by_global_norm(
                    t_list=[
                        tf.math.reduce_mean(input_tensor=tf.stack(BATCH), axis=0)
                        for BATCH in zip(*GRADIENTS_BATCH)
                    ],
                    clip_norm=gradient_clipping_global_norm
                )[0],
                tf.trainable_variables()
            )
        )

        with tf.control_dependencies([VARIABLES_UPDATE]):
            MODEL_UPDATE = EMA_MANAGER.apply(tf.trainable_variables())

            return MODEL_UPDATE


def build_predicts(
        ELMO_MODULE, WORD_EMBEDDING,
        PASSAGE_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
        PASSAGE_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
        PASSAGE_VECTORS_BATCH, QUESTION_VECTORS_BATCH,
        PASSAGE_CONNECTIONS_BATCH, QUESTION_CONNECTIONS_BATCH
):
    def get_greedy_sample(TARGET_DISTRIBUTION):
        return tf.unravel_index(
            indices=tf.dtypes.cast(
                x=tf.math.argmax(
                    tf.reshape(
                        tensor=tf.linalg.band_part(
                            input=tf.linalg.matmul(
                                a=tf.nn.softmax(TARGET_DISTRIBUTION[:1]),
                                b=tf.nn.softmax(TARGET_DISTRIBUTION[1:]),
                                transpose_a=True
                            ),
                            num_lower=tf.dtypes.cast(x=0, dtype=tf.int32),
                            num_upper=tf.math.subtract(
                                x=tf.math.minimum(x=tf.shape(TARGET_DISTRIBUTION)[1], y=answer_span_length_limit),
                                y=1
                            )
                        ),
                        shape=[-1]
                    )
                ),
                dtype=tf.int32
            ),
            dims=[tf.shape(TARGET_DISTRIBUTION)[1], tf.shape(TARGET_DISTRIBUTION)[1]]
        )

    PREDICT_BATCH = []

    for index in range(batch_size):
        with tf.device("/device:GPU:{}".format(index % gpu_count)):
            with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=True):
                ANSWER_SPAN_DISTRIBUTION = feed_forward(
                    ELMO_MODULE, WORD_EMBEDDING,
                    PASSAGE_SYMBOLS_BATCH[index], QUESTION_SYMBOLS_BATCH[index],
                    PASSAGE_NUMBERS_BATCH[index], QUESTION_NUMBERS_BATCH[index],
                    PASSAGE_VECTORS_BATCH[index], QUESTION_VECTORS_BATCH[index],
                    PASSAGE_CONNECTIONS_BATCH[index], QUESTION_CONNECTIONS_BATCH[index],
                    False
                )

                PREDICT_BATCH.append(get_greedy_sample(ANSWER_SPAN_DISTRIBUTION))

    with tf.device("/cpu:0"):
        MODEL_PREDICTS = tf.stack(PREDICT_BATCH)

        return MODEL_PREDICTS


def build_predict(
        ELMO_MODULE, WORD_EMBEDDING,
        PASSAGE_SYMBOLS, QUESTION_SYMBOLS,
        PASSAGE_NUMBERS, QUESTION_NUMBERS,
        PASSAGE_VECTORS, QUESTION_VECTORS,
        PASSAGE_CONNECTIONS, QUESTION_CONNECTIONS
):
    def get_greedy_sample(TARGET_DISTRIBUTION):
        return tf.unravel_index(
            indices=tf.dtypes.cast(
                x=tf.math.argmax(
                    tf.reshape(
                        tensor=tf.linalg.band_part(
                            input=tf.linalg.matmul(
                                a=tf.nn.softmax(TARGET_DISTRIBUTION[:1]),
                                b=tf.nn.softmax(TARGET_DISTRIBUTION[1:]),
                                transpose_a=True
                            ),
                            num_lower=tf.dtypes.cast(x=0, dtype=tf.int32),
                            num_upper=tf.math.subtract(
                                x=tf.math.minimum(x=tf.shape(TARGET_DISTRIBUTION)[1], y=answer_span_length_limit),
                                y=1
                            )
                        ),
                        shape=[-1]
                    )
                ),
                dtype=tf.int32
            ),
            dims=[tf.shape(TARGET_DISTRIBUTION)[1], tf.shape(TARGET_DISTRIBUTION)[1]]
        )

    with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=True):
        ANSWER_SPAN_DISTRIBUTION = feed_forward(
            ELMO_MODULE, WORD_EMBEDDING,
            PASSAGE_SYMBOLS, QUESTION_SYMBOLS,
            PASSAGE_NUMBERS, QUESTION_NUMBERS,
            PASSAGE_VECTORS, QUESTION_VECTORS,
            PASSAGE_CONNECTIONS, QUESTION_CONNECTIONS,
            False
        )

        MODEL_PREDICT = get_greedy_sample(ANSWER_SPAN_DISTRIBUTION)

        return MODEL_PREDICT
