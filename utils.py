import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.metrics import f1_score, classification_report, precision_recall_curve, average_precision_score
import tensorflow.python.ops.numpy_ops.np_config as np_config


np_config.enable_numpy_behavior()
def get_adj(edge_index, node_num):

    row, col = edge_index
    trans = np.concatenate([[col], [row]], axis=0)
    edge_index = np.concatenate([edge_index, trans], axis=-1)

    sparse_adj1 = tf.sparse.SparseTensor(indices=edge_index.T, values=tf.ones(edge_index.shape[1]), dense_shape=[node_num, node_num])
    sparse_adj1 = tf.sparse.reorder(sparse_adj1)

    edge_index1 = sparse_adj1.indices

    self_loop = np.array([range(node_num), range(node_num)])

    return tf.concat([tf.transpose(edge_index1), self_loop], axis=-1), tf.transpose(edge_index1)

#@tf.function
def update_reliability(embedding, triple, source_num, reliability):
    object_index, source_index, claims = triple
    object_num = tf.reduce_max(object_index) + 1
    source_num_dynamic = tf.reduce_max(source_index) + 1
    node_num = object_num + source_num_dynamic

    weights = tf.gather(reliability, source_index)
    object_embedd = tf.gather(embedding, object_index)
    distance = tf.sqrt(tf.reduce_sum(tf.square(object_embedd - claims), axis=-1)) * weights  # * weights tf.sqrt
    weight = tf.square(tf.math.unsorted_segment_sum(data=distance, segment_ids=source_index, num_segments=source_num))
    weight_sum = tf.reduce_sum(weight)
    weight = tf.where(weight_sum > 0, weight / weight_sum, weight)
    r = tf.math.exp(-tf.sqrt(weight))

    r_temp = tf.cast(tf.reshape(r, (-1, 1)), dtype=tf.float32)

    start = tf.cast(object_num, dtype=tf.int32)
    end = tf.cast(node_num, dtype=tf.int32)
    source_indices = tf.range(start, end)
    source_embedd = tf.gather(embedding, source_indices)
    f1 = tf.matmul(source_embedd, source_embedd, transpose_b=True)
    a = tf.reshape(tf.reduce_sum(tf.math.square(source_embedd), axis=-1), (-1, 1))
    f2 = tf.math.sqrt(tf.matmul(a, a, transpose_b=True))
    cor = tf.nn.softmax(f1 / f2)

    reliability = tf.reshape(tf.matmul(cor, r_temp), (tf.shape(r_temp)[0],))

    return reliability

#@tf.function
def update_feature(triple, reliability, object_num, source_num):
    object_index, source_index, claims = triple
    weights = tf.reshape(tf.gather(reliability, source_index), (-1,1))
    # weight = tf.reshape(reliability, (-1,1))
    weighted_claims = claims * weights
    objects_feature = tf.math.unsorted_segment_sum(data=weighted_claims, segment_ids=object_index, num_segments=object_num)
    sources_feature = tf.math.unsorted_segment_sum(data=weighted_claims, segment_ids=source_index, num_segments=source_num)
    sources_feature = tf.where(sources_feature>0.0, x=1.0, y=0.0) * tf.reshape(reliability, shape=(-1,1))
    features = tf.nn.softmax(tf.concat([objects_feature, sources_feature], axis=0))
    return features

# @tf.function
def dis_loss(embedding, triple, reliability, source_num, worker_supervision_mask=None, supervision_boost=0.0):

    object_index, source_index, claims = triple

    object_embedd = tf.gather(embedding, object_index)
    distance1 = tf.sqrt(tf.reduce_sum(tf.square(object_embedd - claims), axis=-1))
    weight1 = tf.math.unsorted_segment_mean(data=distance1, segment_ids=tf.cast(source_index, dtype=tf.int32), num_segments=source_num)
    weighted_reliability = reliability
    if worker_supervision_mask is not None and supervision_boost > 0.0:
        weighted_reliability = reliability * (1.0 + supervision_boost * worker_supervision_mask)
    losses1 = tf.reduce_mean(weight1 * weighted_reliability)

    return losses1

def eval(embedding, truth_set, class_num):
    y_t = truth_set['truths']
    object_index = truth_set['gt_index']
    obj_embedding = tf.gather(embedding, object_index)
    y_p = tf.argmax(tf.nn.softmax(obj_embedding), axis=-1, output_type=tf.int32)
    accuracy_m = keras.metrics.Accuracy()
    accuracy_m.update_state(y_t, y_p)

    return accuracy_m.result().numpy()

def precision_recall_metrics(embedding, truth_set, class_num):
    object_index = truth_set['gt_index']
    y_true = truth_set['truths']
    obj_logits = tf.gather(embedding, object_index)
    probabilities = tf.nn.softmax(obj_logits, axis=-1).numpy()

    metrics = {
        'probabilities': probabilities,
        'labels': y_true
    }

    if class_num == 2:
        positive_scores = probabilities[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_true, positive_scores)
        ap = average_precision_score(y_true, positive_scores)
        metrics.update({
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': ap
        })
    else:
        y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=class_num)
        precision_dict = {}
        recall_dict = {}
        thresholds_dict = {}
        ap_dict = {}
        for cls in range(class_num):
            precision, recall, thresholds = precision_recall_curve(y_true_one_hot[:, cls], probabilities[:, cls])
            precision_dict[cls] = precision
            recall_dict[cls] = recall
            thresholds_dict[cls] = thresholds
            ap_dict[cls] = average_precision_score(y_true_one_hot[:, cls], probabilities[:, cls])
        metrics.update({
            'precision': precision_dict,
            'recall': recall_dict,
            'thresholds': thresholds_dict,
            'average_precision': ap_dict
        })

    return metrics
