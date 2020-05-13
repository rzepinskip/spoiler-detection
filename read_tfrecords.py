import tensorflow as tf
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
max_length = 128


def get_dataset(file_path):
    def read_tfrecord(serialized_example):
        feature_description = {
            "input_ids": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.string),
        }

        example = tf.io.parse_single_example(serialized_example, feature_description)
        input_ids = tf.ensure_shape(
            tf.io.parse_tensor(example["input_ids"], tf.int32), [max_length]
        )
        label = tf.ensure_shape(tf.io.parse_tensor(example["label"], tf.float32), [])

        return {"input_ids": input_ids}, label

    dataset = tf.data.TFRecordDataset(file_path, compression_type="GZIP").map(
        read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return dataset


filename = "GoodreadsSingleGenreAppendedDataset-roberta-128-train.tf.gz"
path = "!bert-hardcoded-separator"
# path = "!roberta-no-sep"
# path = "!roberta-proper-sep"
# path = "!roberta-SEP"
ds = get_dataset(
    f"/home/rzepinskip/Documents/Mgr/!datasets/full-dataset/!roberta-research/{path}/{filename}"
)
x = list(ds.take(20))
tokenizer.decode(x[3][0]["input_ids"])
labels_count = {0.0: 0, 1.0: 0}
for ind, x in ds.enumerate():
    labels_count[x[1].numpy()] += 1


steps_full = (2110317+455921)/512
steps_lite = (1580770+341956)/512