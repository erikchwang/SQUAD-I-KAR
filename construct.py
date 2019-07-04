from utility import *

begin_time = datetime.datetime.now()
ELMO_MODULE = tfh.Module(spec=elmo_url, trainable=False)
WORD_EMBEDDING = tf.Variable(initial_value=load_file(word_embedding_path, "pickle"), trainable=False)
PASSAGE_SYMBOLS_BATCH = [tf.placeholder(dtype=tf.string, shape=[None]) for _ in range(batch_size)]
PASSAGE_NUMBERS_BATCH = [tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size)]
PASSAGE_VECTORS_BATCH = [tf.placeholder(dtype=tf.float32, shape=[None, vector_size]) for _ in range(batch_size)]
PASSAGE_CONNECTIONS_BATCH = [tf.placeholder(dtype=tf.int32, shape=[None, None]) for _ in range(batch_size)]
QUESTION_SYMBOLS_BATCH = [tf.placeholder(dtype=tf.string, shape=[None]) for _ in range(batch_size)]
QUESTION_NUMBERS_BATCH = [tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size)]
QUESTION_VECTORS_BATCH = [tf.placeholder(dtype=tf.float32, shape=[None, vector_size]) for _ in range(batch_size)]
QUESTION_CONNECTIONS_BATCH = [tf.placeholder(dtype=tf.int32, shape=[None, None]) for _ in range(batch_size)]
ANSWER_SPAN_BATCH = [tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size)]
LEARNING_RATE = tf.placeholder(dtype=tf.float32, shape=[])
PASSAGE_SYMBOLS = tf.placeholder(dtype=tf.string, shape=[None])
PASSAGE_NUMBERS = tf.placeholder(dtype=tf.int32, shape=[None])
PASSAGE_VECTORS = tf.placeholder(dtype=tf.float32, shape=[None, vector_size])
PASSAGE_CONNECTIONS = tf.placeholder(dtype=tf.int32, shape=[None, None])
QUESTION_SYMBOLS = tf.placeholder(dtype=tf.string, shape=[None])
QUESTION_NUMBERS = tf.placeholder(dtype=tf.int32, shape=[None])
QUESTION_VECTORS = tf.placeholder(dtype=tf.float32, shape=[None, vector_size])
QUESTION_CONNECTIONS = tf.placeholder(dtype=tf.int32, shape=[None, None])
EMA_MANAGER = tf.train.ExponentialMovingAverage(exponential_moving_average_decay)

MODEL_UPDATE = build_update(
    ELMO_MODULE, WORD_EMBEDDING,
    PASSAGE_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
    PASSAGE_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
    PASSAGE_VECTORS_BATCH, QUESTION_VECTORS_BATCH,
    PASSAGE_CONNECTIONS_BATCH, QUESTION_CONNECTIONS_BATCH,
    ANSWER_SPAN_BATCH, LEARNING_RATE, EMA_MANAGER
)

MODEL_PREDICTS = build_predicts(
    ELMO_MODULE, WORD_EMBEDDING,
    PASSAGE_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
    PASSAGE_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
    PASSAGE_VECTORS_BATCH, QUESTION_VECTORS_BATCH,
    PASSAGE_CONNECTIONS_BATCH, QUESTION_CONNECTIONS_BATCH
)

MODEL_PREDICT = build_predict(
    ELMO_MODULE, WORD_EMBEDDING,
    PASSAGE_SYMBOLS, QUESTION_SYMBOLS,
    PASSAGE_NUMBERS, QUESTION_NUMBERS,
    PASSAGE_VECTORS, QUESTION_VECTORS,
    PASSAGE_CONNECTIONS, QUESTION_CONNECTIONS
)

MODEL_SMOOTH = tf.group(
    *[
        tf.assign(ref=VARIABLE, value=EMA_MANAGER.average(VARIABLE))
        for VARIABLE in tf.trainable_variables()
    ]
)

for index in range(batch_size):
    tf.add_to_collection(name="PASSAGE_SYMBOLS_BATCH", value=PASSAGE_SYMBOLS_BATCH[index])
    tf.add_to_collection(name="PASSAGE_NUMBERS_BATCH", value=PASSAGE_NUMBERS_BATCH[index])
    tf.add_to_collection(name="PASSAGE_VECTORS_BATCH", value=PASSAGE_VECTORS_BATCH[index])
    tf.add_to_collection(name="PASSAGE_CONNECTIONS_BATCH", value=PASSAGE_CONNECTIONS_BATCH[index])
    tf.add_to_collection(name="QUESTION_SYMBOLS_BATCH", value=QUESTION_SYMBOLS_BATCH[index])
    tf.add_to_collection(name="QUESTION_NUMBERS_BATCH", value=QUESTION_NUMBERS_BATCH[index])
    tf.add_to_collection(name="QUESTION_VECTORS_BATCH", value=QUESTION_VECTORS_BATCH[index])
    tf.add_to_collection(name="QUESTION_CONNECTIONS_BATCH", value=QUESTION_CONNECTIONS_BATCH[index])
    tf.add_to_collection(name="ANSWER_SPAN_BATCH", value=ANSWER_SPAN_BATCH[index])

tf.add_to_collection(name="LEARNING_RATE", value=LEARNING_RATE)
tf.add_to_collection(name="PASSAGE_SYMBOLS", value=PASSAGE_SYMBOLS)
tf.add_to_collection(name="PASSAGE_NUMBERS", value=PASSAGE_NUMBERS)
tf.add_to_collection(name="PASSAGE_VECTORS", value=PASSAGE_VECTORS)
tf.add_to_collection(name="PASSAGE_CONNECTIONS", value=PASSAGE_CONNECTIONS)
tf.add_to_collection(name="QUESTION_SYMBOLS", value=QUESTION_SYMBOLS)
tf.add_to_collection(name="QUESTION_NUMBERS", value=QUESTION_NUMBERS)
tf.add_to_collection(name="QUESTION_VECTORS", value=QUESTION_VECTORS)
tf.add_to_collection(name="QUESTION_CONNECTIONS", value=QUESTION_CONNECTIONS)
tf.add_to_collection(name="MODEL_UPDATE", value=MODEL_UPDATE)
tf.add_to_collection(name="MODEL_PREDICTS", value=MODEL_PREDICTS)
tf.add_to_collection(name="MODEL_PREDICT", value=MODEL_PREDICT)
tf.add_to_collection(name="MODEL_SMOOTH", value=MODEL_SMOOTH)
SAVER = tf.train.Saver()
SAVER.export_meta_graph(model_graph_path)

with tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
) as SESSION:
    SESSION.run(tf.initializers.global_variables())
    SAVER.save(sess=SESSION, save_path=model_storage_path, write_meta_graph=False, write_state=False)
    dump_data([], model_progress_path, "json")

print("construct: cost {} seconds".format(int((datetime.datetime.now() - begin_time).total_seconds())))
