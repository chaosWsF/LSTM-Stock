import tensorflow as tf


def lstm_graph(n_steps, n_inputs, n_outputs, n_neurons, n_layers, learning_rate, seed):

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='X_input')
        y = tf.placeholder(tf.float32, [None, n_outputs], name='y_input')
        keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob_input')
        batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size_input')
        tf.add_to_collection('input', x)
        tf.add_to_collection('input', y)
        tf.add_to_collection('input', batch_size)

    with tf.name_scope('weights'):
        w = tf.Variable(tf.truncated_normal([n_neurons, n_outputs], stddev=0.1, seed=seed), dtype=tf.float32, name='W')
        tf.summary.histogram('output_layer_weights', w)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.random_normal([n_outputs], seed=seed), name='b')
        tf.summary.histogram('output_layer_biases', b)

    cells = [tf.contrib.rnn.BasicLSTMCell(n_neurons) for _ in range(n_layers)]
    with tf.name_scope('lstm_dropout'):
        cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, seed=seed) for cell in cells]
    with tf.name_scope('lstm_cells_layers'):
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop, state_is_tuple=True)
    init_state = multi_layer_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, x, initial_state=init_state, dtype=tf.float32)

    with tf.name_scope('output_layer'):
        pred = tf.nn.softmax(tf.matmul(outputs[:, -1, :], w) + b)
        # tf.add_to_collection('prediction', pred)
        tf.summary.histogram('outputs', pred)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), axis=[1]))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.add_to_collection('accuracy', accuracy)
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    
    return x, y, keep_prob, batch_size, training_op, merged
