import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LayerNormalization, Embedding, Conv1D, Add

import transformers


from decision_transformer.models.trajejctory_model import TrajectoryModel
from decision_transformer.models.transformer import GlobalSelfAttention


class SkipConnectWrapper(tf.keras.layers.Layer):
    def __init__(self, inner_layer, *args, **kwargs):
        super().__init__()
        self.inner_layer = inner_layer
        self.adder = Add()

    def call(self, x):
        # print(f'SkipConnectWrapper.call()')
        y = self.inner_layer(x)
        return self.adder([x, y])
    

class GPT2MLP(tf.keras.layers.Layer):
    def __init__(self, n_embed, n_state):
        super().__init__()
        # print(f'GPT2MLP.__init__({n_embed}, {n_state})')
        self.conv1 = Dense(n_state, activation='gelu', name='conv1')
        self.conv2 = Dense(n_embed, name='conv2')

    def call(self, x):
        # print(f'GPT2MLP.call()')
        # print(f'x.shape {x.shape}')
        # print(f'conv1 {self.conv1.__dict__}')
        x = self.conv1(x)
        x = self.conv2(x)
        # print(f'return.shape {x.shape}')
        return x


class GPT2Block(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.layer_norm_1 = LayerNormalization()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        # skip1 = SkipConnectWrapper(Sequential([layer_norm_1, self_attention]))

        self.layer_norm_2 = LayerNormalization()
        self.mlp = GPT2MLP(d_model, dff)
        # skip2 = SkipConnectWrapper(Sequential([layer_norm_2, ffn]))

        # self.block_stack = Sequential([skip1, skip2])



    def call(self, x):
        # print(f'GPT2Block.call()')
        y = self.self_attention(self.layer_norm_1(x))
        x = x+y
        y = self.mlp(self.layer_norm_2(x))
        return x+y
        
        # return self.block_stack(x)


class GPT2Stack(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1, **kwargs ):
        super().__init__()
        # print(f'gpt2stack with {num_layers} layers')
        self.dec_layers = Sequential([
            GPT2Block(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
                for _ in range(num_layers)
            ])
        self.layer_norm_final = LayerNormalization()
        
        

    def call(self, x):
        # print(f'GPT2Stack.call()')
        return self.layer_norm_final(self.dec_layers(x))

class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        print(config)

        cfg = {
            "num_layers": config.n_layer, 
            "d_model": config.n_embd, 
            "num_heads": config.n_head, 
            "dff": config.n_inner,
            "input_vocab_size": config.vocab_size, 
            "target_vocab_size": config.eos_token_id, 
            "dropout_rate":config.attn_pdrop
        }

        
        # Embedding accepts inputs of a max of 
        self.embed_timestep = Embedding(max_ep_len, hidden_size)
        self.embed_return = Dense(hidden_size, name='embed_return')
        self.embed_state = Dense(hidden_size, name='embed_state')
        self.embed_action = Dense(hidden_size, name='embed_action')

        # Double normalization?
        self.embed_ln = LayerNormalization()

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.GPT2_stack = GPT2Stack(**cfg)

        # note: we don't predict states or returns for the paper
        self.predict_state = Dense(self.state_dim, name='predict_state')
        self.predict_action = Dense(self.act_dim, activation='tanh' if action_tanh else '', name='predict_action') 
        self.predict_return = Dense(1, name='predict_return')


    
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = tf.reshape(
                tf.transpose(
                    tf.stack(
                        (returns_embeddings, state_embeddings, action_embeddings), 
                        axis=1), 
                    perm=(0, 2, 1, 3)), 
                (batch_size, 3*seq_length, self.hidden_size)
            )
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = tf.ones((batch_size, seq_length), dtype=tf.float32)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = tf.reshape(
                tf.transpose(
                    tf.stack(
                        (attention_mask, attention_mask, attention_mask), 
                        axis=1),
                    perm=(0, 2, 1)),
                (batch_size, 3*seq_length)
            )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.GPT2_stack(
            stacked_inputs,
            # attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs #['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        x = tf.transpose(
                tf.reshape(x, (batch_size, seq_length, 3, self.hidden_size))
            , perm=(0, 2, 1, 3)
            )

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
