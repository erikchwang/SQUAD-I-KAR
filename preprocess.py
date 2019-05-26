from utility import *

begin_time = datetime.datetime.now()
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument(dest="step_index", type=int)

if argument_parser.parse_args().step_index == 0:
    vocabulary_catalog = [""]
    vocabulary_glove = [[0.0] * glove_size]
    word_count = {}
    word_glove = {}

    for article in load_file(train_dataset_path, "json")["data"] + load_file(develop_dataset_path, "json")["data"]:
        for paragraph in article["paragraphs"]:
            for token in spacy_nlp(paragraph["context"]):
                word_count[token.text] = word_count[token.text] + 1 if token.text in word_count else 1

            for qa in paragraph["qas"]:
                for token in spacy_nlp(qa["question"]):
                    word_count[token.text] = word_count[token.text] + 1 if token.text in word_count else 1

    for item in load_file(glove_archive_path, "text"):
        glove_elements = item.split(" ")

        if glove_elements[0] in word_count and glove_elements[0] not in word_glove:
            word_glove[glove_elements[0]] = [float(element) for element in glove_elements[1:glove_size + 1]]

    for word in sorted(word_count, key=word_count.get, reverse=True):
        if word in word_glove and word not in vocabulary_catalog:
            vocabulary_catalog.append(word)
            vocabulary_glove.append(word_glove[word])

    dump_data(vocabulary_catalog, vocabulary_catalog_path, "text")
    dump_data(vocabulary_glove, vocabulary_glove_path, "json")

elif argument_parser.parse_args().step_index == 1:
    multiprocessing_pool = multiprocessing.Pool()

    train_composite = multiprocessing_pool.map(
        func=enrich_composite,
        iterable=convert_dataset(
            load_file(train_dataset_path, "json"),
            load_file(vocabulary_catalog_path, "text"),
            True
        )
    )

    multiprocessing_pool.close()
    multiprocessing_pool.join()
    dump_data(train_composite, train_composite_path, "json")

elif argument_parser.parse_args().step_index == 2:
    multiprocessing_pool = multiprocessing.Pool()

    develop_composite = multiprocessing_pool.map(
        func=enrich_composite,
        iterable=convert_dataset(
            load_file(develop_dataset_path, "json"),
            load_file(vocabulary_catalog_path, "text"),
            False
        )
    )

    multiprocessing_pool.close()
    multiprocessing_pool.join()
    dump_data(develop_composite, develop_composite_path, "json")

else:
    raise Exception("invalid step index: {}".format(argument_parser.parse_args().step_index))

print(
    "preprocess: cost {} seconds to complete step {}".format(
        int((datetime.datetime.now() - begin_time).total_seconds()),
        argument_parser.parse_args().step_index
    )
)
