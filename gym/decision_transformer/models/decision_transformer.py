import numpy as np
import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


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
            use_cache=False,
            **kwargs
        )
        print(config)
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        # torch.nn.init.ones_(self.embed_timestep.weight)


        self.embed_return = torch.nn.Linear(1, hidden_size)
        # torch.nn.init.ones_(self.embed_return.weight)
        # torch.nn.init.zeros_(self.embed_return.bias)

        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        # torch.nn.init.ones_(self.embed_state.weight)
        # torch.nn.init.zeros_(self.embed_state.bias)

        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        # torch.nn.init.ones_(self.embed_action.weight)
        # torch.nn.init.zeros_(self.embed_action.bias)

        self.embed_ln = nn.LayerNorm(hidden_size)
        # torch.nn.init.ones_(self.embed_ln.weight)
        # torch.nn.init.zeros_(self.embed_ln.bias)


        self.transformer = GPT2Model(config)


        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        # torch.nn.init.ones_(self.predict_state.weight)
        # torch.nn.init.zeros_(self.predict_state.bias)

        self.predict_action = nn.Linear(hidden_size, self.act_dim)
        # torch.nn.init.ones_(self.predict_action.weight)
        # torch.nn.init.zeros_(self.predict_action.bias)
        self.predict_action = nn.Sequential(
            *([self.predict_action] + ([nn.Tanh()] if action_tanh else []))
        )
        
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        # torch.nn.init.ones_(self.predict_return.weight)
        # torch.nn.init.zeros_(self.predict_return.bias)



    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        
        # print(f'forward() returns_to_go.shape {returns_to_go.shape}')
        # print(f'forward() action: {actions}')

        batch_size, seq_length = states.shape[0], states.shape[1]
        # print(f' > forward() batch_size: {batch_size} | seq_len {seq_length}')

        if attention_mask is None:
            assert False
            print('forward() received attention_mask=None')
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        # print(f'states {states}')
        state_embeddings = self.embed_state(states)
        # print(f'state_embeddings {state_embeddings}')
        # print(f'state_embeddings.shape {state_embeddings.shape}')

        # print(f'actions {actions}')
        action_embeddings = self.embed_action(actions)
        # print(f'action_embeddings {action_embeddings}')
        # print(f'action_embeddings.shape {action_embeddings.shape}')

        # print(f'returns_to_go {returns_to_go}')
        returns_embeddings = self.embed_return(returns_to_go)
        # print(f'returns_embeddings {returns_embeddings}')
        # print(f'returns_embeddings.shape {returns_embeddings.shape}')


        time_embeddings = self.embed_timestep(timesteps)
        # print(f'time_embeddings {time_embeddings}')
        

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        # print(f'state_embeddings + time_embeddings {state_embeddings}')
        action_embeddings = action_embeddings + time_embeddings
        # print(f'action_embeddings + time_embeddings {action_embeddings}')
        returns_embeddings = returns_embeddings + time_embeddings
        # print(f'returns_embeddings + time_embeddings {returns_embeddings}')

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        # print(f'forward() stacked_inputs.shape {stacked_inputs.shape}')
        # print(f'forward() stacked_inputs {stacked_inputs}')
        
        #NOTE: restore
        stacked_inputs = self.embed_ln(stacked_inputs)
        # print(f'forward() embedded stacked_inputs {stacked_inputs}')

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        # print(f'forward() stacked_attention_mask.shape {stacked_attention_mask.shape}')

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        # print(f'forward() transformer_outputs["last_hidden_state"].shape {transformer_outputs["last_hidden_state"].shape}')
        x = transformer_outputs['last_hidden_state']
        
        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        # print(f'forward() reshaped transformer_output {x}')

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state
        # print(f'forward() return_preds {return_preds}')
        # print(f'forward() state_preds {state_preds}')
        # print(f'forward() action_preds {action_preds}')
        # assert False
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

            assert timesteps.shape[0] == states.shape[0]
            assert timesteps.shape[0] == actions.shape[0]
            assert timesteps.shape[0] == returns_to_go.shape[0]

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
        # print(f'get_action() input actions {actions.shape}')
        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
