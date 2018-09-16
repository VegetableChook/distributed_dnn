from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
import tensorflow as tf

tf.app.flags.DEFINE_string("ps_hosts", "172.18.0.2:9901", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "172.18.0.3:9902,172.18.0.4:9903",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

# parameters init
lstm_layer = 2
learning_rate = 0.001
training_iters = 5000
batch_size = 128
test_step = 100

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10


def muti_sltm():
    with tf.name_scope("muti_lstm"):
        muti_lstm = rnn.MultiRNNCell([rnn.DropoutWrapper(
            rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True),
            output_keep_prob=0.5) for _ in range(lstm_layer)])
    return muti_lstm


def variable_summaries(var, name):
    # Record that var into a histogram summary
    tf.summary.histogram(name, var)
    # record mean and stddev
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
    tf.summary.scalar('mean/' + name, stddev)


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a new cluster
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create a Server
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    print("Cluster job: %s, task_index: %d, target: %s" % (FLAGS.job_name, FLAGS.task_index, server.target))
    if FLAGS.job_name == "ps":
        server.join()
        print("Parameter Server started")
    elif FLAGS.job_name == "worker":
        # Decide which device to assign task
        with tf.device("/job:worker/task:0"):
            # built half model
            mnist = input_data.read_data_sets("data", one_hot=True)
            with tf.name_scope("input"):
                x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
                y = tf.placeholder(tf.float32, [None, n_classes])
                # input layer
                with tf.name_scope("weights"):
                    l1_weight = tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
                    variable_summaries(l1_weight, "weights")
                with tf.name_scope("biases"):
                    l1_biases = tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]))
                    variable_summaries(l1_biases, "input/biases")
                # hidden layer for input
                with tf.name_scope("Wx_plus_b"):
                    x_ = tf.reshape(x, [-1, n_inputs])
                    x_in = tf.reshape(tf.matmul(x_, l1_weight) + l1_biases, [-1, n_steps, n_hidden_units])
                    tf.summary.histogram("outputs", x_in)
            # cell
            # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
            with tf.name_scope("LSTM"):
                lstm_cell = muti_sltm()
                _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
                outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=_init_state, time_major=False)
                tf.summary.histogram("outputs", outputs)
                tf.summary.histogram("states", states)

        with tf.device("/job:worker/task:1"):
            # an other half model
            with tf.name_scope("output"):
                with tf.name_scope("weights"):
                    l2_weight = tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
                    variable_summaries(l2_weight, "output/weights")
                with tf.name_scope("biases"):
                    l2_biases = tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
                    variable_summaries(l2_biases, "output/biases")
                with tf.name_scope("output"):
                    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
                    results = tf.matmul(outputs[-1], l2_weight) + l2_biases
                    tf.summary.histogram("LSTM/results", results)
            global_step = tf.train.get_or_create_global_step()
            with tf.name_scope("loss"):
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=results, labels=y))
                tf.summary.scalar("loss", cost)
            with tf.name_scope("train"):
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
            # Prediction
            with tf.name_scope("test"):
                with tf.name_scope("correct_prediction"):
                    correct_pred = tf.equal(tf.argmax(results, 1), tf.argmax(y, 1))
                with tf.name_scope("accuracy"):
                    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.summary.scalar("accuracy", accuracy)
            merged = tf.summary.merge_all()

        # create a Supervisor to oversee training process
        hooks = [tf.train.StopAtStepHook(last_step=training_iters)]
        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir="/tensorflow/tem", hooks=hooks) as sess:
            print("Start Worker Session: ", FLAGS.task_index)
            summary_writer = tf.summary.FileWriter("/tensorflow/tensorboard", sess.graph)
            while not sess.should_stop():
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])
                _, step, summary = sess.run([train_op, global_step, merged], feed_dict={x: batch_x, y: batch_y})
                summary_writer.add_summary(summary, step)
                if step % test_step == 0:
                    accuracy_ = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                    print("Task:", FLAGS.task_index, "| step:", step, "| accuracy:", accuracy_)
        summary_writer.close()
        print('Worker Done: ', FLAGS.task_index)


if __name__ == "__main__":
    tf.app.run()
