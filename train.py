#coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)  ##直方图


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')
    lr = tf.Variable(0.001, dtype=tf.float32, name='learning_rate')

with tf.name_scope('layer'):
    with tf.name_scope('Input_layer'):
        with tf.name_scope('W1'):
            W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1), name='W1')
            variable_summaries(W1)
        with tf.name_scope('b1'):
            b1 = tf.Variable(tf.zeros([500]) + 0.1, name='b1')
            variable_summaries(b1)
        with tf.name_scope('L1'):
            L1 = tf.nn.tanh(tf.matmul(x, W1) + b1, name='L1')
    with tf.name_scope('Hidden_layer'):
        with tf.name_scope('W2'):
            W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1), name='W2')
            variable_summaries(W2)
        with tf.name_scope('b2'):
            b2 = tf.Variable(tf.zeros([300]) + 0.1, name='b2')
            variable_summaries(b2)
        with tf.name_scope('L2'):
            L2 = tf.nn.tanh(tf.matmul(L1, W2) + b2, name='L2')
    with tf.name_scope('Output_layer'):
        with tf.name_scope('W3'):
            W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1), name='W3')
            variable_summaries(W3)
        with tf.name_scope('b3'):
            b3 = tf.Variable(tf.zeros([10]) + 0.1, name='b3')
            variable_summaries(b3)
        prediction = tf.nn.softmax(tf.matmul(L2, W3) + b3)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y - prediction))

# 交叉熵代价函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)

# 梯度下降
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.name_scope('train'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

# set config
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.2

with tf.Session(config=tfconfig) as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./graphs/mnist-80', sess.graph)

    for epoch in range(80):
        #sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        sess.run(tf.assign(lr, 0.5e-3))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_xs, y: batch_ys})

        writer.add_summary(summary, epoch)
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        learning_rate = sess.run(lr)
        if epoch % 2 == 0:
            print("Iter" + str(epoch) + ", Testing accuracy:" + str(test_acc) + ", Learning rate:" + str(learning_rate))

    writer.close()
