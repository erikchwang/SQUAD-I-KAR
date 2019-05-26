from utility import *

begin_time = datetime.datetime.now()
ELMO_MODULE = tfh.Module(spec=elmo_module_url, trainable=True)
GLOVE_TABLE = tf.Variable(initial_value=load_file(vocabulary_glove_path, "json"), trainable=False)
PASSAGE_SYMBOLS_BATCH = [tf.placeholder(dtype=tf.string, shape=[None]) for _ in range(batch_size)]
PASSAGE_NUMBERS_BATCH = [tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size)]
PASSAGE_PASSAGE_INDICES_BATCH = [tf.placeholder(dtype=tf.int32, shape=[None, None]) for _ in range(batch_size)]
PASSAGE_PASSAGE_VALUES_BATCH = [tf.placeholder(dtype=tf.bool, shape=[None]) for _ in range(batch_size)]
PASSAGE_PASSAGE_SHAPE_BATCH = [tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size)]
QUESTION_SYMBOLS_BATCH = [tf.placeholder(dtype=tf.string, shape=[None]) for _ in range(batch_size)]
QUESTION_NUMBERS_BATCH = [tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size)]
QUESTION_PASSAGE_INDICES_BATCH = [tf.placeholder(dtype=tf.int32, shape=[None, None]) for _ in range(batch_size)]
QUESTION_PASSAGE_VALUES_BATCH = [tf.placeholder(dtype=tf.bool, shape=[None]) for _ in range(batch_size)]
QUESTION_PASSAGE_SHAPE_BATCH = [tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size)]
ANSWER_SPAN_BATCH = [tf.placeholder(dtype=tf.int32, shape=[None]) for _ in range(batch_size)]
LEARNING_RATE = tf.placeholder(dtype=tf.float32, shape=[])
PASSAGE_SYMBOLS = tf.placeholder(dtype=tf.string, shape=[None])
PASSAGE_NUMBERS = tf.placeholder(dtype=tf.int32, shape=[None])
PASSAGE_PASSAGE_INDICES = tf.placeholder(dtype=tf.int32, shape=[None, None])
PASSAGE_PASSAGE_VALUES = tf.placeholder(dtype=tf.bool, shape=[None])
PASSAGE_PASSAGE_SHAPE = tf.placeholder(dtype=tf.int32, shape=[None])
QUESTION_SYMBOLS = tf.placeholder(dtype=tf.string, shape=[None])
QUESTION_NUMBERS = tf.placeholder(dtype=tf.int32, shape=[None])
QUESTION_PASSAGE_INDICES = tf.placeholder(dtype=tf.int32, shape=[None, None])
QUESTION_PASSAGE_VALUES = tf.placeholder(dtype=tf.bool, shape=[None])
QUESTION_PASSAGE_SHAPE = tf.placeholder(dtype=tf.int32, shape=[None])
EMA_MANAGER = tf.train.ExponentialMovingAverage(ema_manager_decay_rate)

COARSE_UPDATE, FINE_UPDATE = build_update(
    ELMO_MODULE, GLOVE_TABLE,
    PASSAGE_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
    PASSAGE_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
    PASSAGE_PASSAGE_INDICES_BATCH, QUESTION_PASSAGE_INDICES_BATCH,
    PASSAGE_PASSAGE_VALUES_BATCH, QUESTION_PASSAGE_VALUES_BATCH,
    PASSAGE_PASSAGE_SHAPE_BATCH, QUESTION_PASSAGE_SHAPE_BATCH,
    ANSWER_SPAN_BATCH, LEARNING_RATE, EMA_MANAGER
)

MODEL_PREDICTS = build_predicts(
    ELMO_MODULE, GLOVE_TABLE,
    PASSAGE_SYMBOLS_BATCH, QUESTION_SYMBOLS_BATCH,
    PASSAGE_NUMBERS_BATCH, QUESTION_NUMBERS_BATCH,
    PASSAGE_PASSAGE_INDICES_BATCH, QUESTION_PASSAGE_INDICES_BATCH,
    PASSAGE_PASSAGE_VALUES_BATCH, QUESTION_PASSAGE_VALUES_BATCH,
    PASSAGE_PASSAGE_SHAPE_BATCH, QUESTION_PASSAGE_SHAPE_BATCH
)

MODEL_PREDICT = build_predict(
    ELMO_MODULE, GLOVE_TABLE,
    PASSAGE_SYMBOLS, QUESTION_SYMBOLS,
    PASSAGE_NUMBERS, QUESTION_NUMBERS,
    PASSAGE_PASSAGE_INDICES, QUESTION_PASSAGE_INDICES,
    PASSAGE_PASSAGE_VALUES, QUESTION_PASSAGE_VALUES,
    PASSAGE_PASSAGE_SHAPE, QUESTION_PASSAGE_SHAPE
)

MODEL_AVERAGE = tf.group(
    *[
        tf.assign(ref=VARIABLE, value=EMA_MANAGER.average(VARIABLE))
        for VARIABLE in tf.trainable_variables()
    ]
)

for index in range(batch_size):
    tf.add_to_collection(name="PASSAGE_SYMBOLS_BATCH", value=PASSAGE_SYMBOLS_BATCH[index])
    tf.add_to_collection(name="PASSAGE_NUMBERS_BATCH", value=PASSAGE_NUMBERS_BATCH[index])
    tf.add_to_collection(name="PASSAGE_PASSAGE_INDICES_BATCH", value=PASSAGE_PASSAGE_INDICES_BATCH[index])
    tf.add_to_collection(name="PASSAGE_PASSAGE_VALUES_BATCH", value=PASSAGE_PASSAGE_VALUES_BATCH[index])
    tf.add_to_collection(name="PASSAGE_PASSAGE_SHAPE_BATCH", value=PASSAGE_PASSAGE_SHAPE_BATCH[index])
    tf.add_to_collection(name="QUESTION_SYMBOLS_BATCH", value=QUESTION_SYMBOLS_BATCH[index])
    tf.add_to_collection(name="QUESTION_NUMBERS_BATCH", value=QUESTION_NUMBERS_BATCH[index])
    tf.add_to_collection(name="QUESTION_PASSAGE_INDICES_BATCH", value=QUESTION_PASSAGE_INDICES_BATCH[index])
    tf.add_to_collection(name="QUESTION_PASSAGE_VALUES_BATCH", value=QUESTION_PASSAGE_VALUES_BATCH[index])
    tf.add_to_collection(name="QUESTION_PASSAGE_SHAPE_BATCH", value=QUESTION_PASSAGE_SHAPE_BATCH[index])
    tf.add_to_collection(name="ANSWER_SPAN_BATCH", value=ANSWER_SPAN_BATCH[index])

tf.add_to_collection(name="LEARNING_RATE", value=LEARNING_RATE)
tf.add_to_collection(name="PASSAGE_SYMBOLS", value=PASSAGE_SYMBOLS)
tf.add_to_collection(name="PASSAGE_NUMBERS", value=PASSAGE_NUMBERS)
tf.add_to_collection(name="PASSAGE_PASSAGE_INDICES", value=PASSAGE_PASSAGE_INDICES)
tf.add_to_collection(name="PASSAGE_PASSAGE_VALUES", value=PASSAGE_PASSAGE_VALUES)
tf.add_to_collection(name="PASSAGE_PASSAGE_SHAPE", value=PASSAGE_PASSAGE_SHAPE)
tf.add_to_collection(name="QUESTION_SYMBOLS", value=QUESTION_SYMBOLS)
tf.add_to_collection(name="QUESTION_NUMBERS", value=QUESTION_NUMBERS)
tf.add_to_collection(name="QUESTION_PASSAGE_INDICES", value=QUESTION_PASSAGE_INDICES)
tf.add_to_collection(name="QUESTION_PASSAGE_VALUES", value=QUESTION_PASSAGE_VALUES)
tf.add_to_collection(name="QUESTION_PASSAGE_SHAPE", value=QUESTION_PASSAGE_SHAPE)
tf.add_to_collection(name="COARSE_UPDATE", value=COARSE_UPDATE)
tf.add_to_collection(name="FINE_UPDATE", value=FINE_UPDATE)
tf.add_to_collection(name="MODEL_PREDICTS", value=MODEL_PREDICTS)
tf.add_to_collection(name="MODEL_PREDICT", value=MODEL_PREDICT)
tf.add_to_collection(name="MODEL_AVERAGE", value=MODEL_AVERAGE)
SAVER = tf.train.Saver()
SAVER.export_meta_graph(model_graph_path)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as SESSION:
    SESSION.run(tf.global_variables_initializer())
    SAVER.save(sess=SESSION, save_path=model_storage_path, write_meta_graph=False, write_state=False)
    dump_data([], model_progress_path, "json")

print("construct: cost {} seconds".format(int((datetime.datetime.now() - begin_time).total_seconds())))
