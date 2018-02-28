import os
import time

import numpy as np
import tensorflow as tf

from svhn_gan import discriminator, generator
from data import svhn_data
import sys

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 50, "batch size [50]")
flags.DEFINE_string('data_dir', './data/svhn', 'data directory')
flags.DEFINE_string('logdir', './log/svhn', 'log directory')
flags.DEFINE_integer('seed', 324, 'seed ')
flags.DEFINE_integer('seed_data', 631, 'seed data')
flags.DEFINE_integer('labeled', 100, 'labeled data per class')
flags.DEFINE_float('learning_rate', 0.0003, 'learning_rate[0.003]')
flags.DEFINE_float('unl_weight', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_float('lbl_weight', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_float('ma_decay', 0.9999, 'exp moving average for inference [0.9999]')

flags.DEFINE_float('scale', 1e-5, 'scale perturbation')
flags.DEFINE_float('nabla_w', 1e-3, 'weight regularization')
flags.DEFINE_integer('decay_start', 300, 'start of learning rate decay')
flags.DEFINE_integer('epoch', 400, 'labeled data per class')
flags.DEFINE_boolean('nabla', True, 'enable manifold reg')


flags.DEFINE_integer('freq_print', 10000, 'frequency image print tensorboard [10000]')
flags.DEFINE_integer('step_print', 50, 'frequency scalar print tensorboard [50]')
flags.DEFINE_integer('freq_test', 1, 'frequency test [500]')
flags.DEFINE_integer('freq_save', 50, 'frequency saver epoch[50]')

FLAGS = flags.FLAGS
FLAGS._parse_flags()
print("\nParameters")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.lower(), value))
print("")


def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter


def display_progression_epoch(j, id_max):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


def linear_decay(decay_start, decay_end, epoch):
    return min(-1 / (decay_end - decay_start) * epoch + 1 + decay_start / (decay_end - decay_start),1)


def main(_):
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    # Random seed
    rng = np.random.RandomState(FLAGS.seed)  # seed labels
    rng_data = np.random.RandomState(FLAGS.seed_data)  # seed shuffling

    trainx, trainy = svhn_data.load(FLAGS.data_dir, 'train')
    testx, testy = svhn_data.load(FLAGS.data_dir, 'test')
    def rescale(mat):
        return np.transpose(((-127.5 + mat) / 127.5), (3, 0, 1, 2))
    trainx = rescale(trainx)
    testx = rescale(testx)

    trainx_unl = trainx.copy()
    trainx_unl2 = trainx.copy()
    nr_batches_train = int(trainx.shape[0] / FLAGS.batch_size)
    nr_batches_test = int(testx.shape[0] / FLAGS.batch_size)

    # select labeled data
    inds = rng_data.permutation(trainx.shape[0])
    trainx = trainx[inds]
    trainy = trainy[inds]
    txs = []
    tys = []
    for j in range(10):
        txs.append(trainx[trainy == j][:FLAGS.labeled])
        tys.append(trainy[trainy == j][:FLAGS.labeled])
    txs = np.concatenate(txs, axis=0)
    tys = np.concatenate(tys, axis=0)

    print("Data:")
    print('train examples %d, nr batch training %d, test examples %d, nr batch testing %d' \
          % (trainx.shape[0], nr_batches_train, testx.shape[0], nr_batches_test))
    print('histogram train', np.histogram(trainy, bins=10)[0])
    print('histogram test ', np.histogram(testy, bins=10)[0])
    print("histogram labeled", np.histogram(tys, bins=10)[0])
    print("")

    '''construct graph'''
    print('constructing graph')
    unl = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='unlabeled_data_input_pl')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='labeled_data_input_pl')
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input_pl')
    # scalar pl
    lr_pl = tf.placeholder(tf.float32, [], name='learning_rate_pl')
    acc_train_pl = tf.placeholder(tf.float32, [], 'acc_train_pl')
    acc_test_pl = tf.placeholder(tf.float32, [], 'acc_test_pl')
    acc_test_pl_ema = tf.placeholder(tf.float32, [], 'acc_test_pl')
    kl_weight = tf.placeholder(tf.float32, [], 'kl_weight')

    random_z = tf.random_uniform([FLAGS.batch_size, 100], name='random_z')
    perturb = tf.random_normal([FLAGS.batch_size, 100], mean=0, stddev=0.01)
    random_z_pert = random_z + FLAGS.scale * perturb / (
    tf.expand_dims(tf.norm(perturb, axis=1), axis=1) * tf.ones([1, 100]))
    generator(random_z, is_training_pl, init=True)  # init of weightnorm weights cf Salimans et al.
    gen_inp = generator(random_z, is_training_pl, init=False, reuse=True)
    gen_inp_pert = generator(random_z_pert, is_training_pl, init=False, reuse=True)

    discriminator(unl, is_training_pl, init=True)
    logits_lab, _ = discriminator(inp, is_training_pl, init=False,
                                  reuse=True)  # init of weightnorm weights cf Salimans et al.
    logits_gen, layer_fake = discriminator(gen_inp, is_training_pl, init=False, reuse=True)
    logits_unl, layer_real = discriminator(unl, is_training_pl, init=False, reuse=True)
    logits_gen_perturb, layer_fake_perturb = discriminator(gen_inp_pert, is_training_pl, init=False, reuse=True)

    with tf.name_scope('loss_functions'):
        l_unl = tf.reduce_logsumexp(logits_unl, axis=1)
        l_gen = tf.reduce_logsumexp(logits_gen, axis=1)
        # discriminator
        loss_lab = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lbl, logits=logits_lab))
        loss_unl = - 0.5 * tf.reduce_mean(l_unl) \
                   + 0.5 * tf.reduce_mean(tf.nn.softplus(l_unl)) \
                   + 0.5 * tf.reduce_mean(tf.nn.softplus(l_gen))

        # generator
        m1 = tf.reduce_mean(layer_real, axis=0)
        m2 = tf.reduce_mean(layer_fake, axis=0)

        j_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_gen - logits_gen_perturb),axis=1))

        if FLAGS.nabla:
            loss_dis = FLAGS.unl_weight * loss_unl + FLAGS.lbl_weight * loss_lab + kl_weight * j_loss
            loss_gen = tf.reduce_mean(tf.abs(m1 - m2))
            print('manifold reg enabled')
        else:
            loss_dis = FLAGS.unl_weight * loss_unl + FLAGS.lbl_weight * loss_lab
            loss_gen = tf.reduce_mean(tf.abs(m1 - m2))

        correct_pred = tf.equal(tf.cast(tf.argmax(logits_lab, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy_classifier = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()

        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=lr_pl, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_pl, beta1=0.5, name='gen_optimizer')

        with tf.control_dependencies(update_ops_gen):
            train_gen_op = optimizer_gen.minimize(loss_gen, var_list=gvars)

        dis_op = optimizer_dis.minimize(loss_dis, var_list=dvars)

        ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ma_decay)
        maintain_averages_op = ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op)

        logits_ema, _ = discriminator(inp, is_training_pl, getter=get_getter(ema), reuse=True)
        correct_pred_ema = tf.equal(tf.cast(tf.argmax(logits_ema, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy_ema = tf.reduce_mean(tf.cast(correct_pred_ema, tf.float32))

        tf.add_to_collection("logits_lab", logits_lab)
        tf.add_to_collection("logits_ema", logits_ema)
        tf.add_to_collection("correct_op", correct_pred)
        tf.add_to_collection("correct_op_ema", correct_pred_ema)

    with tf.name_scope('summary'):
        with tf.name_scope('discriminator'):
            tf.summary.scalar('loss_discriminator', loss_dis, ['dis'])
            tf.summary.scalar('j_loss', j_loss, ['dis'])

        with tf.name_scope('generator'):
            tf.summary.scalar('loss_generator', loss_gen, ['gen'])

        with tf.name_scope('images'):
            tf.summary.image('gen_images', gen_inp, 10, ['image'])

        with tf.name_scope('epoch'):
            tf.summary.scalar('accuracy_train', acc_train_pl, ['epoch'])
            tf.summary.scalar('accuracy_test_moving_average', acc_test_pl_ema, ['epoch'])
            tf.summary.scalar('accuracy_test_raw', acc_test_pl, ['epoch'])
            tf.summary.scalar('learning_rate', lr_pl, ['epoch'])
            tf.summary.scalar('kl_weight', kl_weight, ['epoch'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')
        sum_op_im = tf.summary.merge_all('image')
        sum_op_epoch = tf.summary.merge_all('epoch')

    # training global varialble
    global_epoch = tf.Variable(0, trainable=False, name='global_epoch')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    inc_global_step = tf.assign(global_step, global_step+1)
    inc_global_epoch = tf.assign(global_epoch, global_epoch+1)

    # op initializer for session manager
    init_gen = [var.initializer for var in gvars][:-3]
    with tf.control_dependencies(init_gen):
        op = tf.global_variables_initializer()
    init_feed_dict = {inp: trainx_unl[:FLAGS.batch_size], unl: trainx_unl[:FLAGS.batch_size],
                                  is_training_pl: True, kl_weight: 0}

    sv = tf.train.Supervisor(logdir=FLAGS.logdir, global_step=global_epoch, summary_op=None, save_model_secs=0,
                             init_op=op,init_feed_dict=init_feed_dict)


    '''////// training //////'''
    print('start training')
    with sv.managed_session() as sess:
        tf.set_random_seed(rng.randint(2 ** 10))
        print('\ninitialization done')
        print('Starting training from epoch :%d, step:%d \n'%(sess.run(global_epoch),sess.run(global_step)))

        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

        while not sv.should_stop():
            epoch = sess.run(global_epoch)
            train_batch = sess.run(global_step)

            if (epoch >= FLAGS.epoch):
                print("Training done")
                sv.stop()
                break

            begin = time.time()
            train_loss_lab= train_loss_unl= train_loss_gen= train_acc= test_acc= test_acc_ma= train_j_loss = 0
            lr = FLAGS.learning_rate * linear_decay(FLAGS.decay_start,FLAGS.epoch,epoch)

            klw = FLAGS.nabla_w

            # construct randomly permuted minibatches
            trainx = []
            trainy = []
            for t in range(int(np.ceil(trainx_unl.shape[0] / float(txs.shape[0])))):  # same size lbl and unlb
                inds = rng.permutation(txs.shape[0])
                trainx.append(txs[inds])
                trainy.append(tys[inds])
            trainx = np.concatenate(trainx, axis=0)
            trainy = np.concatenate(trainy, axis=0)
            trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]  # shuffling unl dataset
            trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                # train discriminator
                feed_dict = {unl: trainx_unl[ran_from:ran_to],
                             is_training_pl: True,
                             inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             lr_pl: lr, kl_weight: klw}
                _, acc, lu, lb, sm = sess.run([train_dis_op, accuracy_classifier, loss_lab, loss_unl, sum_op_dis],
                                                  feed_dict=feed_dict)
                train_loss_unl += lu
                train_loss_lab += lb
                train_acc += acc
                if (train_batch % FLAGS.step_print) == 0:  # to save ram on tensorboard
                    writer.add_summary(sm, train_batch)

                # train generator
                _, lg, sm = sess.run([train_gen_op, loss_gen, sum_op_gen], feed_dict={unl: trainx_unl2[ran_from:ran_to],
                                                                                      is_training_pl: True,
                                                                                      lr_pl: lr, kl_weight: klw})
                train_loss_gen += lg
                if (train_batch % FLAGS.step_print) == 0:
                    writer.add_summary(sm, train_batch)

                if (train_batch % FLAGS.freq_print == 0) & (train_batch != 0):
                    ran_from = np.random.randint(0, trainx_unl.shape[0] - FLAGS.batch_size)
                    ran_to = ran_from + FLAGS.batch_size
                    sm = sess.run(sum_op_im,
                                  feed_dict={is_training_pl: True, unl: trainx_unl[ran_from:ran_to]})
                    writer.add_summary(sm, train_batch)

                train_batch += 1
                sess.run(inc_global_step)

            train_loss_lab /= nr_batches_train
            train_loss_unl /= nr_batches_train
            train_loss_gen /= nr_batches_train
            train_acc /= nr_batches_train
            train_j_loss /= nr_batches_train

            # Testing moving averaged model and raw model after each epoch
            if (epoch % FLAGS.freq_test == 0) | (epoch == 1199):
                idx = rng.permutation(testx.shape[0])
                testx = testx[idx]
                testy = testy[idx]
                for t in range(nr_batches_test):
                    ran_from = t * FLAGS.batch_size
                    ran_to = (t + 1) * FLAGS.batch_size
                    feed_dict = {inp: testx[ran_from:ran_to],
                                 lbl: testy[ran_from:ran_to],
                                 is_training_pl: False}
                    acc, acc_ema = sess.run([accuracy_classifier, accuracy_ema], feed_dict=feed_dict)
                    test_acc += acc
                    test_acc_ma += acc_ema
                test_acc /= nr_batches_test
                test_acc_ma /= nr_batches_test

                sum = sess.run(sum_op_epoch, feed_dict={acc_train_pl: train_acc,
                                                        acc_test_pl: test_acc,
                                                        acc_test_pl_ema: test_acc_ma,
                                                        lr_pl: lr, kl_weight: klw})
                writer.add_summary(sum, epoch)

            print(
                "Epoch %d | time = %ds | kl = %0.2e | lr = %0.2e | loss gen = %.4f | loss lab = %.4f | loss unl = %.4f "
                "| train acc = %.4f| test acc = %.4f | test acc ema = %0.4f"
                % (epoch, time.time() - begin, klw, lr, train_loss_gen, train_loss_lab, train_loss_unl, train_acc,
                   test_acc, test_acc_ma))

            sess.run(inc_global_epoch)

            # save snap shot of model
            if ((epoch % FLAGS.freq_save == 0) & (epoch!=0) ) | (epoch == FLAGS.epoch-1):
                string = 'model-' + str(epoch)
                save_path = os.path.join(FLAGS.logdir, string)
                sv.saver.save(sess, save_path)
                print("Model saved in file: %s" % (save_path))


if __name__ == '__main__':
    tf.app.run()
