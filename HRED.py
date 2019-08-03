# -*- coding:utf-8 -*-

import tensorflow.contrib.seq2seq as tc_seq2seq
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from tensorflow.python.util import nest

from EAttentionWrapper import EAttentionWrapper
from models.model_base import *
from models.model_helper import *
import numpy as np
import grucell_cond

class HREDModel(BaseTFModel):
    # 在类的继承中，，如果重定义某个方法，该方法会覆盖父类的同名方法，
    # 但有时，我们希望能同时实现父类的功能，这时，我们就需要调用父类的方法了，可通过使用 super 来实现
    # 可以实现BaseTFModel中的__init__方法
    def __init__(self, config, mode, scope=None):
        super(HREDModel, self).__init__(config, mode, scope)

    def _build_graph(self):
        self._build_placeholders()   # 定义占位符
        self._build_embeddings()  # 构建词表词向量
        self._build_encoder()   # 搭建编码层（句子gru+上下文lstm）
        self._build_decoder()   # 搭建解码层  lstm + Dense
        if self.mode != ModelMode.infer:
            self._compute_loss()  # 计算loss
            if self.mode == ModelMode.train:
                self.create_optimizer(self.loss)   # 优化调参
                # Training Summary
                self.train_summary = tf.summary.merge([tf.summary.scalar("lr", self.learning_rate),
                                                       tf.summary.scalar("train_loss", self.loss)]
                                                      + self.grad_norm_summary)
        self.saver = tf.train.Saver(tf.global_variables())


    def _build_placeholders(self):
        with tf.variable_scope("placeholders"):
            batch_size = None
            dialog_turn_size = None
            dialog_sent_size = None

            # 上下文输入
            self.source = tf.placeholder(tf.int32,
                                         shape=[batch_size, dialog_turn_size, dialog_sent_size],
                                         name="dialog_inputs")
            # 上下文输入长度
            self.source_length = tf.placeholder(tf.int32,
                                                shape=[batch_size, dialog_turn_size],
                                                name='dialog_input_lengths')
            # 关键字
            self.keyword = tf.placeholder(tf.int32,shape=[batch_size,None],name='keyword_inputs')
            self.keyword_length = tf.placeholder(tf.int32,
                                                shape=[None],
                                                name="keyword_input_lengths")


            # 回复目标输入
            self.target_input = tf.placeholder(tf.int32,
                                               shape=[batch_size, None],
                                               name="response_input_sent")

            # 回复目标输出，相比target_input往后移一位
            self.target_output = tf.placeholder(tf.int32,
                                                shape=[batch_size, None],
                                                name="response_output_sent")
            # 回复输入长度
            self.target_length = tf.placeholder(tf.int32,
                                                shape=[None],
                                                name="target_length")
            # dropout
            self.dropout_keep_prob = tf.placeholder(tf.float32)

            self.batch_size = tf.shape(self.source)[0]
            self.turn_size = tf.shape(self.source)[1]
            self.sent_size = tf.shape(self.source)[2]

            if self.mode != ModelMode.infer:
                self.predict_count = tf.reduce_sum(self.target_length)   # 对target_length求和， reduce_sum()按维度求和

    def _build_embeddings(self):
        with tf.variable_scope("dialog_embeddings"):
            # 词向量
            embedding =np.load('./data/wordEmbedding.npy').astype('float32')
            dialog_embeddings = tf.get_variable("dialog_embeddings",
                                                dtype=tf.float32,
                                                initializer=embedding,
                                                trainable=True)
            print('dialog_embeddings', dialog_embeddings.shape)
            # 编码层与解码层共享词表向量
            self.encoder_embeddings = dialog_embeddings
            self.decoder_embeddings = dialog_embeddings

    def _build_encoder(self):
        with tf.variable_scope("dialog_encoder"):
            with tf.variable_scope('utterance_rnn'):
                # uttn_hidden_size = self.config.emb_size    #  通常，设置隐藏层状态大小与词向量大小一致
                uttn_hidden_size = self.config.emb_size
                uttn_encoder = RNNEncoder(unit_type='gru',
                                          enc_type='uni',
                                          hidden_size=uttn_hidden_size,
                                          num_layers=1,
                                          dropout_keep_prob=self.dropout_keep_prob)
                # encoder_embeddings shape[vocab_size, emb_size]
                # source shape[batch_size, dialog_turn_size, dialog_sent_size]
                # reshape 将source转换成[batch_size*dialog_turn_size, dialog_sent_size]
                uttn_emb_inp = tf.nn.embedding_lookup(self.encoder_embeddings,
                                                      tf.reshape(self.source, [-1, self.sent_size]))
                # [batch_size*dialog_turn_size, dialog_sent_size, emd_size]
                print('utterance input embs shape', uttn_emb_inp.shape)

                uttn_outputs, uttn_states = uttn_encoder(uttn_emb_inp, tf.reshape(self.source_length, [-1]))  # [batch_size*dialog_turn_size]

                uttn_outputs = tf.reshape(uttn_outputs, [self.batch_size,
                                                       -1,
                                                       uttn_hidden_size])

                uttn_states = tf.reshape(uttn_states, [self.batch_size,
                                                       self.turn_size,
                                                       uttn_hidden_size])
                self.uttn_outputs = uttn_outputs
                self.uttn_states = uttn_states
                print('utterance outputs shape', uttn_outputs.shape)
                print('utterance state shape', uttn_states.shape)

            with tf.variable_scope('word_rnn'):
                word_hidden_size = self.config.emb_size
                word_encoder = RNNEncoder(unit_type='gru',
                                          enc_type='uni',
                                          hidden_size=word_hidden_size,
                                          num_layers=1,
                                          dropout_keep_prob=self.dropout_keep_prob)
                word_emb_inp = tf.nn.embedding_lookup(self.encoder_embeddings, self.keyword)
                print('word input embs shape', word_emb_inp.shape)

                word_outputs, word_states = word_encoder(word_emb_inp,self.keyword_length)
                self.word_outputs = word_outputs
                self.word_states = word_states
                print('words_output shape', word_outputs.shape)
                print('words_state shape', word_states.shape)


            with tf.variable_scope("context_rnn"):
                context_encoder = RNNEncoder(unit_type=self.config.unit_type,  #  lstm
                                             enc_type='bi',
                                             hidden_size=self.config.enc_hidden_size,  #
                                             num_layers=self.config.num_layers,
                                             dropout_keep_prob=self.dropout_keep_prob)
                # source_length.shape [batch_size, dialog_turn_size]
                # sign函数 x>0,1;x=0,0;x<0,-1
                # axis=1，对dialog_turn_size求和，shape[batch_size]
                context_turn_length = tf.reduce_sum(tf.sign(self.source_length), axis=1)
                ctx_outputs, ctx_state = context_encoder(uttn_states, context_turn_length)

                self.encoder_outputs = ctx_outputs
                self.encoder_state = ctx_state

    def _build_decoder_cell(self, enc_outputs, enc_state):
        beam_size = self.config.beam_size   # 5,beam search
        context_length = self.source_length
        memory = enc_outputs

        # 预测阶段才进行beam search
        if self.mode == ModelMode.infer and beam_size > 0:
            # beam_search_decoder里面的函数
            enc_state = tc_seq2seq.tile_batch(enc_state,
                                              multiplier=beam_size)

            memory = tc_seq2seq.tile_batch(memory,
                                           multiplier=beam_size)

            context_length = tc_seq2seq.tile_batch(context_length,
                                                   multiplier=beam_size)

        else:
            enc_state = enc_state
            batch_size = self.batch_size

        dec_cell = get_rnn_cell(self.config.unit_type,   # lstm
                                hidden_size=self.config.dec_hidden_size,   # 300
                                num_layers=1,  # 1
                                dropout_keep_prob=self.dropout_keep_prob)

        return dec_cell, enc_state

    def _build_decoder(self):
        with tf.variable_scope("dialog_decoder"):
            with tf.variable_scope("decoder_output_projection"):   # 全连接层
                output_layer = layers_core.Dense(
                    self.config.vocab_size, use_bias=False, name="output_projection")  # units单元个数  词表大小

            with tf.variable_scope("decoder_rnn"):
                attn_mech = tc_seq2seq.BahdanauAttention(self.config.dec_hidden_size, self.word_outputs, None)
                attn_mech1 = tc_seq2seq.BahdanauAttention(self.config.dec_hidden_size,self.uttn_outputs,None)
                attn_mech2 = tc_seq2seq.BahdanauAttention(self.config.dec_hidden_size, self.encoder_outputs,None)

                self.att1 = attn_mech.batch_size
                self.att2 = attn_mech.batch_size
                self.att3 = attn_mech.batch_size

                dec_cell = GRUCell(self.config.dec_hidden_size)

                #dec_cell = grucell_cond.GRUCellCond(self.config.dec_hidden_size)
                #self.encoder_outputs = tf.reshape(self.encoder_outputs,[-1,self.config.dec_hidden_size*2])
                #dec_cell = grucell_cond.CondWrapper(dec_cell, self.encoder_outputs)
                #word_outputs = tf.reshape(self.word_outputs,[self.batch_size,-1])

                dec_cell = EAttentionWrapper(dec_cell, [attn_mech, attn_mech1, attn_mech2], attention_layer_size=[self.config.dec_hidden_size,self.config.dec_hidden_size,self.config.dec_hidden_size] )
                #print('self.batch_size',self.batch_size)
                dec_init_state = dec_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

                # Training or Eval
                if self.mode != ModelMode.infer:  # not infer, do decode turn by turn

                    resp_emb_inp = tf.nn.embedding_lookup(self.decoder_embeddings, self.target_input)
                    helper = tc_seq2seq.TrainingHelper(resp_emb_inp, self.target_length)
                    decoder = tc_seq2seq.BasicDecoder(
                        cell=dec_cell,
                        helper=helper,
                        initial_state=dec_init_state,    # 编码层的最终状态
                        output_layer=output_layer   # 全连接层
                    )

                    dec_outputs, dec_state, _ = tc_seq2seq.dynamic_decode(decoder)
                    sample_id = dec_outputs.sample_id
                    logits = dec_outputs.rnn_output

                else:
                    start_tokens = tf.fill([self.batch_size], self.config.sos_idx)
                    end_token = self.config.eos_idx
                    maximum_iterations = tf.to_int32(self.config.infer_max_len)


                    helper = tc_seq2seq.GreedyEmbeddingHelper(self.decoder_embeddings,
                                                  start_tokens=start_tokens,
                                                  end_token=tf.constant(end_token, dtype=tf.int32))

                    decoder = tc_seq2seq.BasicDecoder(
                        cell=dec_cell,
                        helper=helper,
                        initial_state=dec_init_state,
                        output_layer=output_layer  # 全连接层
                    )

                    dec_outputs, dec_state, _ = tc_seq2seq.dynamic_decode(decoder,maximum_iterations=maximum_iterations)
                    logits = tf.no_op()
                    sample_id = dec_outputs.sample_id

                self.logits = logits
                self.sample_id = sample_id

    def _compute_loss(self):
        with tf.variable_scope('loss'):
            """Compute optimization loss."""

            batch_size = tf.shape(self.target_output)[0]
            # 最大长度
            max_time = tf.shape(self.target_output)[1]
            #print('logits target',self.logits.shape,self.target_output.shape)
            #print('tf.shape(self.logits)[1], max_len',tf.shape(self.logits)[1], self.config.max_len)

            output_maxlen = tf.minimum(tf.shape(self.target_output)[1], self.config.max_len)
            #output_maxlen = tf.minimum(tf.shape(self.logits)[1], self.config.max_len)
            out_data_slice = tf.slice(self.target_output,[0,0],[-1,output_maxlen])
            out_logits_slice = tf.slice(self.logits,[0,0,0],[-1,output_maxlen,-1])

            # 损失函数交叉熵
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=out_data_slice, logits=out_logits_slice)  # 真实值与预测值
            target_weights = tf.sequence_mask(self.target_length+self.keyword_length, maxlen=output_maxlen, dtype=self.logits.dtype)
            loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(batch_size)  #  loss公式?
            self.loss = loss
            self.exp_ppl = tf.reduce_sum(crossent * target_weights)
        pass

    def train(self, sess, batch_input):
        assert self.mode == ModelMode.train
        feed_dict = {
            self.source: batch_input.source,
            self.source_length: batch_input.source_length,
            self.keyword:batch_input.keyword,
            self.keyword_length:batch_input.keyword_length,
            self.target_input: batch_input.target_input,
            self.target_output: batch_input.target_output,
            self.target_length: batch_input.target_length,
            self.dropout_keep_prob: self.config.dropout_keep_prob
        }


        #print('word_outputs.shape[0]',self.word_outputs.shape[0])
        #print('self.keyword_length.shape',self.keyword_length.shape)


        res = sess.run([self.keyword_length,
                        self.source_length,
                        self.update_opt,
                        self.loss,
                        self.exp_ppl,
                        self.predict_count,
                        self.batch_size,
                        self.train_summary,
                        self.global_step], feed_dict)
        return res[1:]
        pass

    def eval(self, sess, batch_input):
        assert self.mode == ModelMode.eval
        #print([d.shape for d in batch_input])
        feed_dict = {
            self.source: batch_input.source,
            self.source_length: batch_input.source_length,
            self.keyword: batch_input.keyword,
            self.keyword_length: batch_input.keyword_length,
            self.target_input: batch_input.target_input,
            self.target_output: batch_input.target_output,
            self.target_length: batch_input.target_length,
            self.dropout_keep_prob: 1.0
        }
        res = sess.run([self.loss,
                        self.exp_ppl,
                        self.predict_count,
                        self.batch_size,
                        self.global_step], feed_dict)
        return res

    def infer(self, sess, batch_data):
        print(batch_data.source.shape)
        print(batch_data.source_length.shape)
        print(batch_data.keyword.shape)
        print(batch_data.keyword_length.shape)
        assert self.mode == ModelMode.infer
        feed_dict = {
            self.source: batch_data.source,
            self.source_length: batch_data.source_length,
            self.keyword: batch_data.keyword,
            self.keyword_length: batch_data.keyword_length,
            self.dropout_keep_prob: 1.0
        }

        res = sess.run([self.sample_id,
                        self.batch_size], feed_dict)

        return res
