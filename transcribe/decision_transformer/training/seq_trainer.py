import numpy as np
import tensorflow as tf

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # action_target = torch.clone(actions)
        with tf.GradientTape() as tape:
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
            )

            act_dim = action_preds.shape[2]
            action_preds = tf.reshape(action_preds, (-1, act_dim))[tf.reshape(attention_mask, (-1)) > 0]
            actions = tf.reshape(actions, (-1, act_dim))[tf.reshape(attention_mask,(-1)) > 0]

            loss = self.loss_fn(
                None, action_preds, None,
                None, actions, None,
            )

            # self.optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            # self.optimizer.step()

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # with torch.no_grad():
        #     self.diagnostics['training/action_error'] = tf.reduce_mean((action_preds-actions)**2).detach().cpu().item()

        self.diagnostics['training/action_error'] = tf.reduce_mean((action_preds-actions)**2)
        return loss
        # return loss.detach().cpu().item()
